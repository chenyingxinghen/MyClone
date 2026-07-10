"""
MyClone - Process chat history into training data for fine-tuning.
Supports WeChat (WeChatMsg CSV) and QQ (NapCat JSON) exports.

Pipeline:
  1. Load WeChat CSVs and QQ JSONs from raw export directory
  2. Filter skip types, system messages, blocked words
  3. Group consecutive same-sender messages (time window)
  4. Generate multi-turn QA pairs in ShareGPT format
  5. Save to dataset/sft/ for LLaMA-Factory

Usage:
  pip install pandas
  python process_data.py

Dependencies: pandas (that's it)
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def _safe_print(msg: str):
    try:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    except Exception:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


try:
    import pandas as pd
except ImportError:
    _safe_print("Missing dependency: pip install pandas")
    sys.exit(1)


# ── Load config ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("data", {})
    return {}


_cfg = load_config()

RAW_DATA_DIR = Path(_cfg.get("raw_data_dir", "../WeClone/my-chat-history"))
OUTPUT_DIR = Path(_cfg.get("output_dir", "./dataset"))
SYSTEM_PROMPT = _cfg.get("system_prompt", "请你扮演一名人类，不要说自己是人工智能")
COMBINE_WINDOW = _cfg.get("combine_time_window", 10) * 60
QA_WINDOW = _cfg.get("qa_match_time_window", 10) * 60
MAX_MSG_LEN = _cfg.get("max_message_length", 2048)
MIN_TURNS = _cfg.get("min_turns", 2)
MIN_CHARS = _cfg.get("min_total_chars", 4)
# Cap how many consecutive messages get glued into a single turn. Without this,
# a burst of connected messages becomes one giant multi-line reply and the model
# learns to fire off run-on message walls. 4 keeps replies natural.
MAX_COMBINE = _cfg.get("max_combine_messages", 4)

# Qwen3/Qwen3.5 are reasoning models: at inference they emit a <think>...</think>
# block before the reply. Chat data has no chain-of-thought, so we bake an EMPTY
# think block into every assistant turn. The model then learns "after the
# assistant header, emit an empty think, then reply" — thinking is stably OFF and
# the reply matches your style. Toggle off for non-reasoning base models.
ADD_EMPTY_THINK = _cfg.get("add_empty_think", True)
EMPTY_THINK = "<think>\n\n</think>\n\n"

# Aggregated diagnostics, printed at the end so nothing is dropped silently.
DROP_STATS = {"burst_msgs_dropped": 0, "bursts_capped": 0}
EXCLUDE_CONTACTS = set(_cfg.get("exclude_contacts", ["文件传输助手"]))
BLOCKED_WORDS: list = _cfg.get("blocked_words", [])
QQ_SELF_QQ: int = _cfg.get("qq_self_qq", 0)
CONTACT_RELATIONS: dict = _cfg.get("contact_relations", {})

# ── Type mappings ───────────────────────────────────────────────────────────

WX_TYPE_MAP = {
    "text": "文本", "image": "图片", "video": "视频", "voice": "语音",
    "file": "文件", "sticker": "动画表情", "sticker2": "动画表情",
    "location": "位置", "system": "系统通知", "share": "(分享)卡片式链接",
    "card link": "(分享)卡片式链接", "music": "(分享)音乐",
    "transfer": "转账", "voice call": "语音通话", "pat pat": "拍一拍",
    "merged forward chat records": "合并转发的聊天记录",
    "reply with quote": "引用回复", "message recall": "消息撤回",
    "add friend": "添加好友",
}

CUT_TYPES = {
    "图片", "视频", "合并转发的聊天记录", "语音", "Cut",
    "(分享)音乐", "(分享)卡片式链接", "(分享)笔记", "(分享)小程序",
    "(分享)收藏夹", "(分享)视频号名片", "(分享)视频号视频",
    "粘贴的文本", "未知",
}

SKIP_TYPES = {
    "添加好友", "推荐公众号", "动画表情", "用户上传的GIF表情",
    "位置", "文件", "位置共享", "引用回复", "群公告",
    "转账", "语音通话", "系统通知", "消息撤回", "拍一拍", "邀请加群",
}

SYSTEM_MSG_RE = re.compile("|".join([
    r'\[转账[收付款]?\]\s*￥[\d.]+',
    r'\[收款\]\s*￥[\d.]+',
    r'\[QQ红包\]',
    r'你已添加了.*?(?:现在可以开始聊天|以上是打招呼)',
    r'刚刚把你添加到通讯录',
    r'以上是打招呼的(?:消息|内容)',
]))

QQ_TYPE_MAP = {
    (2, 1): "文本", (2, 2): "图片", (2, 3): "图片",
    (2, 16): "动画表情", (2, 17): "文本", (2, 129): "文本",
    (2, 1025): "文本", (2, 4096): "动画表情", (2, 8194): "图片",
    (3, 1): "文件", (3, 2): "文件", (3, 4): "文件",
    (5, 1): "消息撤回", (5, 4): "消息撤回",
    (6, 0): "语音", (7, 0): "视频",
    (9, 33): "(分享)卡片式链接", (9, 34): "(分享)卡片式链接",
    (9, 48): "(分享)卡片式链接", (9, 49): "(分享)卡片式链接",
    (10, 0): "系统通知", (11, 0): "系统通知", (11, 7): "系统通知",
    (17, 8): "拍一拍",
    (19, 1): "语音通话", (19, 3): "语音通话", (19, 5): "语音通话",
    (1, 1): "消息撤回", (1, 2): "消息撤回",
    (23, 0): "系统通知", (24, 1): "系统通知", (25, 1): "系统通知",
}


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    type_name: str
    is_sender: int
    talker: str
    msg: str
    create_time: datetime


@dataclass
class CutMessage:
    create_time: datetime


# ── Relation loading ────────────────────────────────────────────────────────

def load_existing_relations() -> dict:
    """Scan WeClone's dataset/csv/ for existing users.json relation files."""
    csv_dir = RAW_DATA_DIR.parent / "dataset" / "csv"
    if not csv_dir.exists():
        return {}
    relations = {}
    for folder in csv_dir.iterdir():
        if not folder.is_dir():
            continue
        users_file = folder / "users.json"
        if users_file.exists():
            try:
                data = json.loads(users_file.read_text("utf-8"))
                rel = data.get("relation", "")
                if rel:
                    name = folder.name
                    if name.startswith("wx_"):
                        name = name[3:]
                    elif name.startswith("qq_"):
                        name = name[3:]
                    relations[name] = rel
            except (json.JSONDecodeError, OSError):
                pass
    return relations


# ── Source loaders ──────────────────────────────────────────────────────────

def load_wechat_contact(
    contact_dir: Path,
    relations: dict,
) -> Tuple[List[Union[ChatMessage, CutMessage]], str]:
    """Load one WeChat contact folder → (messages, relation)."""
    csv_files = [f for f in contact_dir.iterdir() if f.suffix == ".csv"]
    if not csv_files:
        return [], ""

    contact_name = contact_dir.name.replace("私聊_", "").strip()
    relation = relations.get(contact_name, "")

    all_messages: List[Union[ChatMessage, CutMessage]] = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, dtype={"msg": str, "src": str}, keep_default_na=False)

        if "type_name" in df.columns:
            df["type_name"] = df["type_name"].map(lambda x: WX_TYPE_MAP.get(x, x))

        df = df[~df["type_name"].isin(SKIP_TYPES)]

        if "is_forward" in df.columns:
            df = df[~((df["is_sender"] == 1) & (df["is_forward"].astype(bool)))]

        df["CreateTime"] = pd.to_datetime(df["CreateTime"], utc=True)
        df = df.reset_index(drop=True)

        is_text = df["type_name"] == "文本"
        drop = is_text & df["msg"].str.contains(SYSTEM_MSG_RE, na=False)
        drop = drop | (is_text & (df["msg"].str.strip() == ""))
        if BLOCKED_WORDS:
            bp = "|".join(re.escape(w) for w in BLOCKED_WORDS)
            drop = drop | (is_text & df["msg"].str.contains(bp, regex=True, na=False))
        df = df[~drop]

        for _, row in df.iterrows():
            tn = row["type_name"]
            ct = row["CreateTime"].to_pydatetime()

            if tn in CUT_TYPES:
                all_messages.append(CutMessage(create_time=ct))
            elif tn == "文本" and str(row.get("msg", "")).strip():
                all_messages.append(ChatMessage(
                    type_name="文本",
                    is_sender=int(row["is_sender"]),
                    talker=str(row.get("talker", "")),
                    msg=str(row["msg"]).strip(),
                    create_time=ct,
                ))

    return all_messages, relation


def load_qq_chat(
    json_path: Path,
    self_qq: int,
    relations: dict,
) -> Tuple[List[Union[ChatMessage, CutMessage]], str]:
    """Parse NapCat QQ JSON → (messages, relation)."""
    CST = timezone(timedelta(hours=8))

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw.sort(key=lambda m: m.get("msgTime", 0))

    qq_num = re.search(r"(\d+)", json_path.stem)
    contact_key = qq_num.group(1) if qq_num else json_path.stem
    relation = relations.get(contact_key, "")

    messages: List[Union[ChatMessage, CutMessage]] = []

    for msg in raw:
        key = (msg.get("msgType", 0), msg.get("subMsgTypepb", 0))
        tn = QQ_TYPE_MAP.get(key)
        if tn is None or tn in SKIP_TYPES:
            continue

        sender_qq = msg.get("senderQQNum", 0)
        is_sender = 1 if sender_qq == self_qq else 0

        parts = [
            e.strip() for e in msg.get("elements", [])
            if isinstance(e, str) and e.strip()
            and e not in ("[Emoji表情]", "[动画表情]", "[不支持的消息]")
        ]
        content = "".join(parts).strip()

        try:
            ct = datetime.fromtimestamp(msg.get("msgTime", 0), tz=CST)
        except (OSError, ValueError, OverflowError):
            ct = datetime(1970, 1, 1, tzinfo=timezone.utc)

        talker = msg.get("sendNickName", "") or str(sender_qq)

        if tn in CUT_TYPES:
            messages.append(CutMessage(create_time=ct))
        elif tn == "文本" and content:
            messages.append(ChatMessage(
                type_name="文本", is_sender=is_sender,
                talker=talker, msg=content, create_time=ct,
            ))

    return messages, relation


# ── Processing pipeline ────────────────────────────────────────────────────

def time_close(a: datetime, b: datetime, window: float) -> bool:
    try:
        return abs((b - a).total_seconds()) <= window
    except Exception:
        return False


def group_consecutive(
    messages: List[Union[ChatMessage, CutMessage]],
) -> List[Union[ChatMessage, CutMessage]]:
    """Group consecutive same-sender messages within time window."""
    if not messages:
        return []

    def combine(group: List[ChatMessage]) -> ChatMessage:
        # Cap the run at MAX_COMBINE messages so a long burst becomes a natural
        # short reply, not a wall of glued lines. Overflow is counted (see
        # DROP_STATS) rather than dropped silently.
        if len(group) > MAX_COMBINE:
            DROP_STATS["burst_msgs_dropped"] += len(group) - MAX_COMBINE
            DROP_STATS["bursts_capped"] += 1
            group = group[:MAX_COMBINE]
        base = group[0]
        text = base.msg
        for m in group[1:]:
            if not m.msg:
                continue
            if text and text[-1] not in "。.！!？?…，,":
                text += "\n"
            text += m.msg
        if len(text) > MAX_MSG_LEN:
            text = text[:MAX_MSG_LEN]
        return ChatMessage(
            type_name=base.type_name, is_sender=base.is_sender,
            talker=base.talker, msg=text, create_time=group[-1].create_time,
        )

    result: List[Union[ChatMessage, CutMessage]] = []
    group: List[ChatMessage] = []

    for msg in messages:
        if isinstance(msg, CutMessage):
            if group:
                result.append(combine(group) if len(group) > 1 else group[0])
                group = []
            result.append(msg)
            continue

        if not msg.msg:
            continue

        if not group:
            group = [msg]
            continue

        last = group[-1]
        if (msg.is_sender == last.is_sender
                and msg.talker == last.talker
                and time_close(last.create_time, msg.create_time, COMBINE_WINDOW)):
            group.append(msg)
        else:
            result.append(combine(group) if len(group) > 1 else group[0])
            group = [msg]

    if group:
        result.append(combine(group) if len(group) > 1 else group[0])

    return result


def generate_qa(
    messages: List[Union[ChatMessage, CutMessage]],
    relation: str,
) -> List[dict]:
    """Build strictly alternating user→assistant conversations.

    Uses two buffers (user side, assistant side). A completed (user, assistant)
    pair is only committed when the *next* user message arrives, at a time gap,
    or at a CutMessage/flush. This guarantees:
      * every conversation alternates user, assistant, user, assistant, ...
      * a real question always precedes its answer (no "answer→question" skew)
      * self-initiated bursts get a synthetic <begin_chat> user turn
      * no turn exceeds MAX_COMBINE glued messages
    """
    results: List[dict] = []
    conv: List[dict] = []      # committed, strictly alternating turns
    ubuf: List[str] = []       # buffered consecutive other-person (user) msgs
    abuf: List[str] = []       # buffered consecutive self (assistant) msgs
    last_time: Optional[datetime] = None

    def buf_lines(buf: List[str]) -> int:
        return sum(s.count("\n") + 1 for s in buf)

    def add(buf: List[str], text: str):
        """Append to a turn buffer, hard-capping total lines at MAX_COMBINE."""
        if not text:
            return
        avail = MAX_COMBINE - buf_lines(buf)
        if avail <= 0:
            return
        lines = text.split("\n")
        buf.append("\n".join(lines[:avail]))

    def commit_pair():
        """Freeze the current user+assistant buffers into one Q&A pair."""
        if abuf:  # only a real answer yields a training pair
            user_text = "\n".join(ubuf) if ubuf else "<begin_chat>"
            asst_text = "\n".join(abuf)
            if len(user_text) > MAX_MSG_LEN:
                user_text = user_text[:MAX_MSG_LEN]
            if len(asst_text) > MAX_MSG_LEN:
                asst_text = asst_text[:MAX_MSG_LEN]
            conv.append({"role": "user", "content": user_text})
            conv.append({"role": "assistant", "content": asst_text})
        ubuf.clear()
        abuf.clear()

    def flush():
        nonlocal conv
        commit_pair()
        try:
            if len(conv) < MIN_TURNS:
                return
            total = sum(len(m["content"]) for m in conv)
            if total < MIN_CHARS or total > MAX_MSG_LEN:
                return
            # A lone self-initiated pair with no real context isn't useful.
            if (len(conv) == 2 and conv[0]["role"] == "user"
                    and conv[0]["content"] == "<begin_chat>"):
                return

            sys_prompt = SYSTEM_PROMPT
            if relation:
                sys_prompt += f"\n 对方是你的{relation}，你们正在聊天"

            processed = list(conv)
            for i in range(len(processed) - 1):
                if (processed[i]["role"] == "user"
                        and "<begin_chat>" in processed[i]["content"]
                        and processed[i + 1]["role"] == "assistant"):
                    hint = processed[i + 1]["content"]
                    processed[i] = {
                        "role": "user",
                        "content": processed[i]["content"].replace(
                            "<begin_chat>",
                            f"<begin_chat>你应该说：{hint}</begin_chat>",
                        ),
                    }

            # Bake an empty think block into each assistant turn — done last so
            # the <begin_chat> hint above stays free of think tags and the char
            # filters above measure real content, not the boilerplate prefix.
            if ADD_EMPTY_THINK:
                processed = [
                    {"role": m["role"], "content": EMPTY_THINK + m["content"]}
                    if m["role"] == "assistant" else m
                    for m in processed
                ]

            results.append({
                "time": last_time.isoformat() if last_time else None,
                "messages": processed,
                "images": [],
                "system": sys_prompt,
            })
        finally:
            conv = []

    for msg in messages:
        if isinstance(msg, CutMessage):
            flush()
            last_time = None
            continue

        # A long silence ends the current conversation.
        if last_time is not None and not time_close(last_time, msg.create_time, QA_WINDOW):
            flush()
            last_time = None

        if msg.is_sender == 0:  # other person → user turn
            if abuf:            # previous answer is done; a new question begins
                commit_pair()
            add(ubuf, msg.msg)
        else:                   # self → assistant turn
            add(abuf, msg.msg)

        last_time = msg.create_time

    flush()
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    _safe_print(f"Raw data: {RAW_DATA_DIR.resolve()}")
    _safe_print(f"Output:   {OUTPUT_DIR.resolve()}")

    # Build relation map: existing users.json → config overrides
    relations = load_existing_relations()
    relations.update(CONTACT_RELATIONS)
    if relations:
        _safe_print(f"Relations: {len(relations)} contacts")

    all_qa: List[dict] = []

    # Process WeChat contacts
    wx_dir = RAW_DATA_DIR / "wxchat" / "texts"
    if wx_dir.exists():
        contacts = sorted([d for d in wx_dir.iterdir() if d.is_dir()])
        _safe_print(f"\nWeChat: {len(contacts)} contacts")
        for contact_dir in contacts:
            name = contact_dir.name.replace("私聊_", "")
            if name in EXCLUDE_CONTACTS or contact_dir.name in EXCLUDE_CONTACTS:
                continue

            messages, relation = load_wechat_contact(contact_dir, relations)
            if not messages:
                continue

            grouped = group_consecutive(messages)
            qa_pairs = generate_qa(grouped, relation)
            _safe_print(f"  {name}: {len(messages)} msgs -> {len(grouped)} grouped -> {len(qa_pairs)} QA")
            all_qa.extend(qa_pairs)

    # Process QQ chats
    qq_dir = RAW_DATA_DIR / "qqchat"
    if qq_dir.exists() and QQ_SELF_QQ:
        _safe_print(f"\nQQ: self_qq={QQ_SELF_QQ}")
        for f in sorted(qq_dir.iterdir()):
            if f.suffix == ".json":
                messages, relation = load_qq_chat(f, QQ_SELF_QQ, relations)
                if not messages:
                    continue
                grouped = group_consecutive(messages)
                qa_pairs = generate_qa(grouped, relation)
                _safe_print(f"  {f.stem}: {len(messages)} msgs -> {len(qa_pairs)} QA")
                all_qa.extend(qa_pairs)

    # Assign IDs and save
    for i, qa in enumerate(all_qa):
        qa["id"] = str(i)
        qa["score"] = 0

    sft_dir = OUTPUT_DIR / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    output_path = sft_dir / "sft-my.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=4)

    dataset_info = {
        "chat-sft": {
            "file_name": "sft-my.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "system": "system"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    with open(sft_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    # Stats
    _safe_print(f"\n{'=' * 50}")
    _safe_print(f"Total QA pairs: {len(all_qa)}")
    if all_qa:
        turns = [len(q["messages"]) for q in all_qa]
        chars = [sum(len(m["content"]) for m in q["messages"]) for q in all_qa]
        _safe_print(f"  Turns: min={min(turns)}, max={max(turns)}, avg={sum(turns)/len(turns):.1f}")
        _safe_print(f"  Chars: min={min(chars)}, max={max(chars)}, avg={sum(chars)/len(chars):.0f}")

        sys_dist: Dict[str, int] = {}
        for qa in all_qa:
            sys_dist[qa["system"]] = sys_dist.get(qa["system"], 0) + 1
        _safe_print("  System prompts:")
        for sp, cnt in sorted(sys_dist.items(), key=lambda x: -x[1]):
            _safe_print(f"    [{cnt}] {sp[:80]}")

    _safe_print(f"\nOutput: {output_path}")
    if DROP_STATS["bursts_capped"]:
        _safe_print(
            f"Burst cap (MAX_COMBINE={MAX_COMBINE}): "
            f"{DROP_STATS['bursts_capped']} runs trimmed, "
            f"{DROP_STATS['burst_msgs_dropped']} overflow messages dropped."
        )
    _safe_print("Upload dataset/ folder to Kaggle for training.")


if __name__ == "__main__":
    main()
