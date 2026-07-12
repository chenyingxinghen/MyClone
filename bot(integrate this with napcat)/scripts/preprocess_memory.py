"""将原始聊天整理为有认知价值的完整会话，不删除原始消息。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOT_DIR = Path(__file__).resolve().parents[1]

from components.memory_store import MemoryStore

RULES_VERSION = 2


GENERIC = {
    "嗯", "恩", "哦", "噢", "昂", "啊", "额", "呃", "好", "好的", "行", "可以",
    "知道了", "没事", "没", "是", "不是", "对", "不对", "哈哈", "哈哈哈", "嘿嘿",
    "笑死", "6", "666", "？", "?", "。", "…", "。。。", "晚安", "早安",
}
SYSTEM_PATTERNS = (
    # QQ 关系和客户端系统提示。
    r"请求添加你为好友",
    r"已成为好友",
    r"你已添加了.*现在可以开始聊天",
    r"撤回了一条消息",
    # 导出器无法还原实际内容时留下的占位符，中英文方括号均兼容。
    r"^[\[【]?(图片|图片消息|表情|表情消息|emoji表情|动画表情|语音|语音消息|"
    r"视频|视频消息|应用消息|文件|位置|分享)[\]】]?$",
    # 音视频通话状态没有可供人格认知抽取的文本语义。
    r"^[\[【]?(语音通话|视频通话)[\]】]?$",
    r"^(语音|视频)?通话(时长|已结束|结束|已取消|取消|未接通|对方已拒绝|无人接听).*$",
    r"^通话时长[：:]?.*$",
)


@dataclass
class Item:
    msg_id: str
    sender_id: str
    sent_at: int
    text: str
    informative: bool


def normalize(text: str) -> str:
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def is_system(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(re.search(pattern, compact, re.I) for pattern in SYSTEM_PATTERNS)


def is_informative(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).lower()
    if not compact or compact in GENERIC or re.fullmatch(r"[哈呵嘿]{1,8}", compact):
        return False
    if re.fullmatch(r"https?://\S+", compact):
        return False
    semantic = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact)
    if len(semantic) < 3:
        return False
    # 问句、叙述句和带时间/数字的信息，即使较短也可能有认知价值。
    return len(semantic) >= 6 or bool(re.search(r"[？?！!]|为什么|怎么|觉得|喜欢|讨厌|想|要|会|因为|今天|明天|昨天|\d", compact))


def raw_sessions(rows, gap_seconds: int):
    current, last = [], None
    for row in rows:
        text = normalize(row["text"])
        if not text or is_system(text):
            continue
        if current and row["sent_at"] - last > gap_seconds:
            yield current
            current = []
        # 连续重复发送只保留一份。
        if current and current[-1].sender_id == row["sender_id"] and current[-1].text == text \
                and row["sent_at"] - current[-1].sent_at <= 120:
            last = row["sent_at"]
            continue
        current.append(Item(str(row["msg_id"]), str(row["sender_id"]),
                            int(row["sent_at"]), text, is_informative(text)))
        last = row["sent_at"]
    if current:
        yield current


def split_large(session: list[Item], max_chars: int):
    current, size = [], 0
    for item in session:
        line_size = len(item.text) + 32
        if current and size + line_size > max_chars:
            yield current
            # 两条重叠消息帮助保留切分边界的语境。
            current = current[-2:]
            size = sum(len(x.text) + 32 for x in current)
        current.append(item)
        size += line_size
    if current:
        yield current


def assess(part: list[Item]) -> tuple[float, bool, str]:
    informative = [x for x in part if x.informative]
    speakers = {x.sender_id for x in part}
    semantic_chars = sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", x.text)) for x in informative)
    longest = max((len(x.text) for x in informative), default=0)
    turns = sum(a.sender_id != b.sender_id for a, b in zip(part, part[1:]))
    score = min(100.0, len(informative) * 2.5 + min(semantic_chars, 300) / 10
                + min(turns, 12) * 1.5 + (8 if len(speakers) >= 2 else 0))
    keep = (len(informative) >= 3 and semantic_chars >= 24 and len(speakers) >= 2) \
        or (len(informative) >= 2 and semantic_chars >= 45) \
        or longest >= 80
    if keep:
        reason = "meaningful_dialogue" if len(speakers) >= 2 else "substantial_monologue"
    elif not informative:
        reason = "only_short_reactions"
    elif len(speakers) < 2:
        reason = "isolated_message"
    else:
        reason = "low_information_density"
    return round(score, 2), keep, reason


def main():
    p = argparse.ArgumentParser(description="预处理聊天并筛选具有长期认知价值的会话")
    p.add_argument("--db", type=Path, default=BOT_DIR / "data" / "memory.db")
    p.add_argument("--peer", required=True)
    p.add_argument("--gap-minutes", type=int, default=30)
    p.add_argument("--max-chars", type=int, default=7000)
    p.add_argument("--prune-placeholders", action="store_true",
                   help="从工作数据库删除明确的占位符/系统消息；原始 JSON 不受影响")
    args = p.parse_args()
    store = MemoryStore(args.db)
    with store.connect() as db:
        rows = db.execute("SELECT id,msg_id,sender_id,sent_at,text FROM messages "
                          "WHERE peer_id=? ORDER BY sent_at,id", (args.peer,)).fetchall()
        source_count = len(rows)
        excluded_ids = [row["id"] for row in rows if is_system(normalize(row["text"]))]
        if args.prune_placeholders and excluded_ids:
            db.executemany("DELETE FROM messages WHERE id=?", ((x,) for x in excluded_ids))
            rows = db.execute("SELECT id,msg_id,sender_id,sent_at,text FROM messages "
                              "WHERE peer_id=? ORDER BY sent_at,id", (args.peer,)).fetchall()
        db.execute("DELETE FROM prepared_conversations WHERE peer_id=?", (args.peer,))
        prepared = []
        for session in raw_sessions(rows, args.gap_minutes * 60):
            for part in split_large(session, args.max_chars):
                score, keep, reason = assess(part)
                ids = [x.msg_id for x in part]
                key_data = f"{args.peer}:{ids[0]}:{ids[-1]}:{len(ids)}"
                key = hashlib.sha256(key_data.encode()).hexdigest()[:24]
                prepared.append((key, args.peer, part[0].sent_at, part[-1].sent_at,
                                 json.dumps(ids), len(part), sum(x.informative for x in part),
                                 len({x.sender_id for x in part}), score, int(keep), reason,
                                 int(time.time())))
        db.executemany("""INSERT INTO prepared_conversations
          (chunk_key,peer_id,start_time,end_time,message_ids,message_count,
           informative_count,speaker_count,quality_score,keep,reason,prepared_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", prepared)
        stats = db.execute("""SELECT count(*) total,sum(keep) kept,
          sum(CASE WHEN keep=1 THEN message_count ELSE 0 END) kept_messages,
          round(avg(CASE WHEN keep=1 THEN quality_score END),2) avg_quality
          FROM prepared_conversations WHERE peer_id=?""", (args.peer,)).fetchone()
        reasons = db.execute("""SELECT reason,count(*) n FROM prepared_conversations
          WHERE peer_id=? GROUP BY reason ORDER BY n DESC""", (args.peer,)).fetchall()
        db.execute("""INSERT INTO preprocessing_runs
          (peer_id,run_at,source_messages,removed_placeholders,prepared_chunks,kept_chunks,rules_version)
          VALUES(?,?,?,?,?,?,?)""", (args.peer, int(time.time()), source_count,
          len(excluded_ids) if args.prune_placeholders else 0, stats["total"], stats["kept"], RULES_VERSION))
    print(f"预处理完成：{stats['total']} 段，保留 {stats['kept']} 段，"
          f"覆盖 {stats['kept_messages']} 条上下文消息，平均质量 {stats['avg_quality']}")
    if args.prune_placeholders:
        print(f"已从工作数据库清除 {len(excluded_ids)} 条占位符/系统消息（原始 JSON 保持不变）")
    print("筛选分布：" + "，".join(f"{x['reason']}={x['n']}" for x in reasons))


if __name__ == "__main__":
    main()
