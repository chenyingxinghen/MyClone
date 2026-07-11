"""Validate MyClone SFT data without importing the GPU training stack."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


THINK_RE = re.compile(r"^<think>\s*</think>\s*")
MEDIA_PLACEHOLDER_RE = re.compile(
    r"\[(?:图片|视频|语音|文件|动画表情|Emoji表情|不支持的消息)\]"
)
BRACKET_ARTIFACT_RE = re.compile(r"\[[^\]\n]{1,240}\]")
MENTION_LINE_RE = re.compile(r"^\s*@(?:所有人|\S+)\s*$")
URL_RE = re.compile(r"(?:https?://|www\.|ur\.alipay\.com/)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
SENSITIVE_KEYWORD_RE = re.compile(
    r"(?:账号密码|密码|验证码|身份证|银行卡|支付口令|access[_ -]?key|api[_ -]?key|secret|token)",
    re.IGNORECASE,
)


def contains_sensitive_text(text: str) -> bool:
    return bool(URL_RE.search(text) or EMAIL_RE.search(text)
                or LONG_NUMBER_RE.search(text) or SENSITIVE_KEYWORD_RE.search(text))


def plain_reply(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def resolve_dataset(path: str) -> tuple[Path, Path]:
    candidate = Path(path).expanduser().resolve()
    data_file = candidate / "sft-my.json" if candidate.is_dir() else candidate
    return data_file, data_file.parent


def validate(data_file: Path, dataset_dir: Path) -> tuple[dict, list[str]]:
    rows = json.loads(data_file.read_text(encoding="utf-8"))
    errors: list[str] = []
    reply_counts: Counter[str] = Counter()
    signatures = set()
    duplicate_rows = 0
    pair_count = 0
    answer_leaks = 0
    media_placeholders = 0
    bracket_artifacts = 0
    mention_lines = 0
    sensitive_messages = 0
    copied_user_replies = 0
    invalid_rows = 0
    empty_replies = 0
    reply_lengths = []
    multiline_replies = 0
    sources = set()

    for row in rows:
        messages = row.get("messages", [])
        sources.add(row.get("source", ""))
        if (not messages or len(messages) % 2
                or any(msg.get("role") != ("user" if i % 2 == 0 else "assistant")
                       for i, msg in enumerate(messages))):
            invalid_rows += 1

        signature = hashlib.sha256(json.dumps(
            {"messages": messages, "system": row.get("system", "")},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")).hexdigest()
        if signature in signatures:
            duplicate_rows += 1
        signatures.add(signature)

        for i in range(0, len(messages) - 1, 2):
            pair_count += 1
            user = messages[i].get("content", "")
            reply = plain_reply(messages[i + 1].get("content", ""))
            answer_leaks += "<begin_chat>你应该说：" in user
            media_placeholders += bool(MEDIA_PLACEHOLDER_RE.search(reply))
            bracket_artifacts += len(BRACKET_ARTIFACT_RE.findall(user))
            bracket_artifacts += len(BRACKET_ARTIFACT_RE.findall(reply))
            mention_lines += sum(
                bool(MENTION_LINE_RE.fullmatch(line.strip()))
                for text in (user, reply) for line in text.splitlines()
            )
            sensitive_messages += contains_sensitive_text(user)
            sensitive_messages += contains_sensitive_text(reply)
            copied_user_replies += bool(len(user) >= 4 and user in reply)
            empty_replies += not reply
            if reply:
                reply_counts[reply] += 1
                reply_lengths.append(len(reply))
                multiline_replies += "\n" in reply

    quality_path = dataset_dir / "quality-report.json"
    quality = json.loads(quality_path.read_text("utf-8")) if quality_path.exists() else {}
    max_short_chars = int(quality.get("config", {}).get("max_short_reply_chars", 4))
    short_cap = int(quality.get("config", {}).get("max_short_reply_occurrences", 30))
    short_counts = {reply: count for reply, count in reply_counts.items()
                    if len(reply) <= max_short_chars}
    max_short_count = max(short_counts.values(), default=0)

    if invalid_rows:
        errors.append(f"{invalid_rows} conversations have invalid user/assistant alternation")
    if answer_leaks:
        errors.append(f"{answer_leaks} user turns contain their target answer")
    if media_placeholders:
        errors.append(f"{media_placeholders} assistant replies contain media placeholders")
    if bracket_artifacts:
        errors.append(f"{bracket_artifacts} exporter-style square-bracket artifacts remain")
    if mention_lines:
        errors.append(f"{mention_lines} standalone @ mention lines remain")
    if sensitive_messages:
        errors.append(f"{sensitive_messages} messages may contain URLs, credentials or identifiers")
    if copied_user_replies:
        errors.append(f"{copied_user_replies} assistant replies copy the full user message")
    if duplicate_rows:
        errors.append(f"{duplicate_rows} exact duplicate conversations remain")
    if empty_replies:
        errors.append(f"{empty_replies} assistant replies are empty")
    if short_cap > 0 and max_short_count > short_cap:
        errors.append(
            f"a short reply occurs {max_short_count} times, above configured cap {short_cap}"
        )
    if quality:
        if quality.get("conversations") != len(rows):
            errors.append("quality-report conversation count does not match sft-my.json")
        if quality.get("qa_pairs_after_rebalance") != pair_count:
            errors.append("quality-report pair count does not match sft-my.json")

    summary = {
        "dataset": str(data_file),
        "conversations": len(rows),
        "qa_pairs": pair_count,
        "sources": len(sources - {""}),
        "invalid_conversations": invalid_rows,
        "answer_leaks": answer_leaks,
        "assistant_media_placeholders": media_placeholders,
        "square_bracket_artifacts": bracket_artifacts,
        "standalone_mention_lines": mention_lines,
        "potential_sensitive_messages": sensitive_messages,
        "assistant_copies_user": copied_user_replies,
        "exact_duplicate_conversations": duplicate_rows,
        "empty_assistant_replies": empty_replies,
        "unique_assistant_ratio": round(len(reply_counts) / pair_count, 4) if pair_count else 0,
        "assistant_reply_le_5_ratio": round(
            sum(length <= 5 for length in reply_lengths) / len(reply_lengths), 4
        ) if reply_lengths else 0,
        "multiline_assistant_replies": multiline_replies,
        "multiline_assistant_ratio": round(
            multiline_replies / len(reply_lengths), 4
        ) if reply_lengths else 0,
        "max_identical_short_reply_count": max_short_count,
        "configured_short_reply_cap": short_cap,
        "passed": not errors,
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/dataset/sft",
                        help="Directory containing sft-my.json, or the JSON file itself")
    parser.add_argument("--output", default="",
                        help="Optional path for the JSON validation report")
    args = parser.parse_args()

    data_file, dataset_dir = resolve_dataset(args.dataset)
    if not data_file.exists():
        print(f"Dataset not found: {data_file}", file=sys.stderr)
        return 2

    summary, errors = validate(data_file, dataset_dir)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
