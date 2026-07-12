"""从已导入的历史消息中分段抽取长期认知；可反复运行并断点续跑。"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOT_DIR = Path(__file__).resolve().parents[1]

from components.memory_extraction import (PROMPT, RETRY_SUFFIX, parse_json_array,
                                          request_extraction, salvage_json_array)
from components.memory_store import MemoryStore


async def run(args):
    store = MemoryStore(args.db)
    with store.connect() as db:
        prepared = db.execute("""SELECT * FROM prepared_conversations
          WHERE peer_id=? AND keep=1 ORDER BY start_time,chunk_key""", (args.partner,)).fetchall()
        if not prepared:
            raise RuntimeError("没有预处理结果，请先运行 scripts/preprocess_memory.py")
    root = args.api_base.rstrip("/")
    host = (urlparse(root).hostname or "").lower()
    is_ollama = host in {"localhost", "127.0.0.1", "::1"}
    if is_ollama and root.endswith("/v1"):
        root = root[:-3]
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    done = partial = failed = saved = attempted = 0
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for index, prepared_part in enumerate(prepared):
            if index < args.start or (args.limit and attempted >= args.limit):
                continue
            ids = json.loads(prepared_part["message_ids"])
            marks = ",".join("?" for _ in ids)
            with store.connect() as db:
                rows = db.execute(f"SELECT msg_id,sender_id,sent_at,text FROM messages WHERE msg_id IN ({marks})", ids).fetchall()
            by_id = {row["msg_id"]: row for row in rows}
            part = [(by_id[mid], f"[{by_id[mid]['sent_at']}] QQ{by_id[mid]['sender_id']}: {by_id[mid]['text']}")
                    for mid in ids if mid in by_id]
            key = prepared_part["chunk_key"]
            with store.connect() as db:
                old = db.execute("SELECT status FROM extraction_jobs WHERE chunk_key=?", (key,)).fetchone()
                if old and old["status"] == "done":
                    continue
                db.execute("""INSERT INTO extraction_jobs
                  (chunk_key,start_time,end_time,message_ids,status,attempts,updated_at)
                  VALUES(?,?,?,?, 'running',1,strftime('%s','now'))
                  ON CONFLICT(chunk_key) DO UPDATE SET status='running',attempts=attempts+1,
                  updated_at=strftime('%s','now')""",
                  (key, prepared_part["start_time"], prepared_part["end_time"], json.dumps(ids)))
            prompt = PROMPT.format(self_id=args.self_id, partner_id=args.partner,
                                   max_items=args.max_items,
                                   dialogue="\n".join(x[1] for x in part))
            attempted += 1
            try:
                url = root + ("/api/chat" if is_ollama else "/chat/completions")
                best_partial: list = []
                parse_error = None
                finish_reason = "unknown"
                retried = False
                for request_index in range(args.json_retries + 1):
                    request_prompt = prompt
                    if request_index:
                        retried = True
                        request_prompt += RETRY_SUFFIX.format(max_items=args.max_items)
                    content, finish_reason = await request_extraction(
                        client, url=url, headers=headers, prompt=request_prompt,
                        model=args.model, is_ollama=is_ollama,
                        max_output_tokens=args.max_output_tokens)
                    try:
                        items = parse_json_array(content)
                        parse_error = None
                        break
                    except (ValueError, json.JSONDecodeError) as exc:
                        parse_error = exc
                        repaired = salvage_json_array(content)
                        if len(repaired) > len(best_partial):
                            best_partial = repaired
                else:
                    items = best_partial
                is_partial = parse_error is not None
                if is_partial and not items:
                    raise parse_error
                items = items[:args.max_items]
                relation_id = f"relationship:{args.self_id}:{args.partner}"
                normalized = []
                for item in items:
                    if not isinstance(item, dict) or item.get("kind") not in {"self", "person", "relationship", "episode"}:
                        continue
                    item["subject_id"] = ({"self": args.self_id, "person": args.partner}
                                          .get(item["kind"], relation_id))
                    item["confidence"] = min(.95, max(.05, float(item.get("confidence", .7))))
                    normalized.append(item)
                items = normalized
                saved += store.save_memories(items, ids)
                with store.connect() as db:
                    if is_partial:
                        error = (f"partial JSON salvaged: {len(items)} items; "
                                 f"finish_reason={finish_reason}; error={parse_error!r}")
                        db.execute("UPDATE extraction_jobs SET status='partial',error=?,updated_at=strftime('%s','now') WHERE chunk_key=?",
                                   (error[:1000], key))
                    else:
                        db.execute("UPDATE extraction_jobs SET status='done',error=NULL,updated_at=strftime('%s','now') WHERE chunk_key=?", (key,))
                if is_partial:
                    partial += 1
                    print(f"[{index}] 部分修复，保留 {len(items)} 条，结束原因 {finish_reason}")
                else:
                    done += 1
                    retry_note = "（重试后成功）" if retried else ""
                    print(f"[{index}] 完成{retry_note}，认知 {len(items)} 条：{items}")
            except Exception as exc:
                failed += 1
                with store.connect() as db:
                    db.execute("UPDATE extraction_jobs SET status='failed',error=?,updated_at=strftime('%s','now') WHERE chunk_key=?", (repr(exc)[:1000], key))
                print(f"[{index}] 失败：{type(exc).__name__}: {exc}")
    print(f"抽取结束：完成 {done}，部分修复 {partial}，失败 {failed}，写入/更新 {saved}")


def main():
    load_dotenv(BOT_DIR / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=BOT_DIR / "data" / "memory.db")
    p.add_argument("--self-id", default=3512024837)
    p.add_argument("--partner", default=1422105979)
    p.add_argument("--api-base", default=os.getenv("LLM_API_BASE", "http://127.0.0.1:11434"))
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen3:8b"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout", type=float, default=360)
    p.add_argument("--max-items", type=int, default=12)
    p.add_argument("--max-output-tokens", type=int, default=2600)
    p.add_argument("--json-retries", type=int, default=1)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
