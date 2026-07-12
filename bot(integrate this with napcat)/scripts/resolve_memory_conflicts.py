"""检测并回写记忆冲突；任务可断点续跑，重复执行不会重复处理已完成的 pair。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOT_DIR = Path(__file__).resolve().parents[1]

from components.embedding_client import is_local_ollama, ollama_root
from components.memory_store import MemoryStore


PROMPT = """你在维护一个人的长期记忆。请判断下面两条记忆之间的事实关系。
只输出一个紧凑 JSON 对象，不要 Markdown、解释或思考过程。

relation 必须是以下之一：
- duplicate：同一事实的重复表述
- supports：新证据支持并加强同一事实
- supersedes：一个较新的状态替代另一个旧状态
- contradicts：两条事实在同一有效时间内互斥，但证据不足以判断谁取代谁
- coexists：相关但可以同时成立，或描述不同时间阶段
- unrelated：不是同一个事实

winner 必须是 left、right 或 none。duplicate/supports 选择应保留的规范记忆；
supersedes 选择代表较新有效状态的记忆；其他关系通常为 none。
不要仅因发生时间不同就认定冲突，也不要用“更新的一条总是正确”作为规则。

输出字段：relation, winner, confidence(0到1), reason。

left:
{left}

right:
{right}
"""

BATCH_PROMPT = """你在维护一个人的长期记忆。请判断输入数组中每一对记忆的事实关系。
只输出一个紧凑 JSON 数组，不要 Markdown、解释、理由或思考过程，必须为每个 pair_index 输出一项。

relation 必须是 duplicate、supports、supersedes、contradicts、coexists、unrelated 之一：
- duplicate：同一事实的重复表述
- supports：新证据支持并加强同一事实
- supersedes：一个较新的状态替代另一个旧状态
- contradicts：同一有效时间内互斥，但证据不足以判断谁取代谁
- coexists：相关但可以同时成立，或描述不同时间阶段
- unrelated：不是同一个事实

winner 必须是 left、right 或 none。duplicate/supports 选择应保留的规范记忆；
supersedes 选择较新的有效状态；其他关系通常为 none。
不要仅因发生时间不同就认定冲突，也不要假设更新的一条总是正确。

每项只输出字段：pair_index, relation, winner, confidence(0到1)。

输入：
{pairs}
"""


RELATIONS = {"duplicate", "supports", "supersedes", "contradicts",
             "coexists", "unrelated"}


def memory_payload(row) -> str:
    evidence = json.loads(row["evidence_json"] or "[]")
    return json.dumps({
        "id": int(row["id"]),
        "kind": row["kind"],
        "subject_id": row["subject_id"],
        "summary": row["summary"],
        "keywords": row["keywords"],
        "confidence": row["confidence"],
        "valid_from": row["valid_from"],
        "valid_to": row["valid_to"],
        "evidence_count": len(evidence),
    }, ensure_ascii=False)


def parse_object(content: str) -> dict:
    content = content.strip()
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    if "```" in content:
        blocks = content.split("```")
        content = next((x.removeprefix("json").strip()
                        for x in blocks[1::2] if "{" in x), content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型响应中没有 JSON 对象")
    result = json.loads(content[start:end + 1])
    if result.get("relation") not in RELATIONS:
        raise ValueError(f"无效 relation: {result.get('relation')!r}")
    if result.get("winner") not in {"left", "right", "none"}:
        raise ValueError(f"无效 winner: {result.get('winner')!r}")
    result["confidence"] = min(1.0, max(0.0, float(result.get("confidence", 0))))
    result["reason"] = str(result.get("reason", ""))[:1000]
    return result


def parse_array(content: str) -> list[dict]:
    content = content.strip()
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    if "```" in content:
        blocks = content.split("```")
        content = next((x.removeprefix("json").strip()
                        for x in blocks[1::2] if "[" in x), content)
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end < start:
        raise ValueError("模型响应中没有 JSON 数组")
    items = json.loads(content[start:end + 1])
    if not isinstance(items, list):
        raise ValueError("模型响应不是 JSON 数组")
    normalized = []
    for result in items:
        if not isinstance(result, dict) or result.get("relation") not in RELATIONS:
            continue
        if result.get("winner") not in {"left", "right", "none"}:
            continue
        try:
            result["pair_index"] = int(result.get("pair_index"))
        except (TypeError, ValueError):
            continue
        result["confidence"] = min(1.0, max(0.0, float(result.get("confidence", 0))))
        result["reason"] = f"模型判定为 {result['relation']}"
        normalized.append(result)
    return normalized


def prepare_jobs(store: MemoryStore, model: str, top_k: int,
                 min_similarity: float) -> int:
    with store.connect() as db:
        rows = db.execute("""SELECT m.*,e.dimensions,e.embedding FROM memories m
          JOIN memory_embeddings e ON e.memory_id=m.id
          WHERE e.model=? AND m.status IN ('active','disputed')
          ORDER BY m.subject_id,m.kind,m.id""", (model,)).fetchall()
        if not rows:
            raise RuntimeError(f"没有模型 {model} 的记忆向量，请先运行 scripts/backfill_embeddings.py")

        groups: dict[tuple[str, str], list] = {}
        for row in rows:
            groups.setdefault((row["subject_id"], row["kind"]), []).append(row)
        jobs: set[tuple] = set()
        now = int(time.time())
        for group_rows in groups.values():
            valid_rows = []
            vectors = []
            for row in group_rows:
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
                if vector.size != int(row["dimensions"]):
                    continue
                valid_rows.append(row)
                vectors.append(vector)
            if len(vectors) < 2:
                continue
            matrix = np.stack(vectors)
            similarities = matrix @ matrix.T
            for index, left in enumerate(valid_rows):
                left_id = int(left["id"])
                candidate_indexes = np.flatnonzero(similarities[index] >= min_similarity)
                candidate_indexes = [i for i in candidate_indexes
                                     if int(valid_rows[i]["id"]) > left_id]
                candidate_indexes.sort(key=lambda i: float(similarities[index, i]), reverse=True)
                for right_index in candidate_indexes[:top_k]:
                    right_id = int(valid_rows[right_index]["id"])
                    pair_key = hashlib.sha256(f"{left_id}:{right_id}".encode()).hexdigest()[:24]
                    jobs.add((pair_key, left_id, right_id, "pending", now))
        before = db.execute("SELECT count(*) FROM conflict_jobs").fetchone()[0]
        db.executemany("""INSERT OR IGNORE INTO conflict_jobs
          (pair_key,left_memory_id,right_memory_id,status,updated_at)
          VALUES(?,?,?,?,?)""", jobs)
        after = db.execute("SELECT count(*) FROM conflict_jobs").fetchone()[0]
        return after - before


async def judge(client: httpx.AsyncClient, args, left, right) -> dict:
    local = is_local_ollama(args.api_base)
    root = ollama_root(args.api_base) if local else args.api_base.rstrip("/")
    url = f"{root}/api/chat" if local else f"{root}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    prompt = PROMPT.format(left=memory_payload(left), right=memory_payload(right))
    payload = {"model": args.model, "stream": False,
               "messages": [{"role": "user", "content": prompt}]}
    if local:
        payload.update({"think": False, "format": "json", "keep_alive": args.keep_alive,
                        "options": {"temperature": .05, "num_predict": 500}})
    else:
        payload.update({"temperature": .05, "max_tokens": 500})
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    content = (data.get("message", {}).get("content", "") if local else
               data.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return parse_object(content)


async def judge_batch(client: httpx.AsyncClient, args, pairs: list[tuple]) -> list[dict]:
    local = is_local_ollama(args.api_base)
    root = ollama_root(args.api_base) if local else args.api_base.rstrip("/")
    url = f"{root}/api/chat" if local else f"{root}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    pair_payload = [{
        "pair_index": index,
        "left": json.loads(memory_payload(left)),
        "right": json.loads(memory_payload(right)),
    } for index, (job, left, right) in enumerate(pairs)]
    prompt = BATCH_PROMPT.format(pairs=json.dumps(pair_payload, ensure_ascii=False))
    payload = {"model": args.model, "stream": False,
               "messages": [{"role": "user", "content": prompt}]}
    if local:
        payload.update({"think": False, "format": "json", "keep_alive": args.keep_alive,
                        "options": {"temperature": .05,
                                    "num_predict": max(300, len(pairs) * 80)}})
    else:
        payload.update({"temperature": .05, "max_tokens": max(300, len(pairs) * 80)})
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    content = (data.get("message", {}).get("content", "") if local else
               data.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return parse_array(content)


def choose_canonical(left, right, winner: str):
    if winner == "left":
        return left, right
    if winner == "right":
        return right, left
    ordered = sorted((left, right), key=lambda row: (
        float(row["confidence"]), int(row["valid_from"] or 0), int(row["id"])
    ), reverse=True)
    return ordered[0], ordered[1]


def merge_evidence(left: str, right: str) -> str:
    values = list(dict.fromkeys(json.loads(left or "[]") + json.loads(right or "[]")))
    return json.dumps(values, ensure_ascii=False)


def apply_result(db, job, left, right, result: dict, apply_threshold: float) -> None:
    now = int(time.time())
    relation = result["relation"]
    confidence = result["confidence"]
    winner_id = None
    if confidence >= apply_threshold:
        if relation in {"duplicate", "supports"}:
            canonical, redundant = choose_canonical(left, right, result["winner"])
            winner_id = int(canonical["id"])
            evidence = merge_evidence(canonical["evidence_json"], redundant["evidence_json"])
            merged_confidence = min(.95, max(float(canonical["confidence"]),
                                             float(redundant["confidence"])))
            db.execute("UPDATE memories SET evidence_json=?,confidence=?,updated_at=? WHERE id=?",
                       (evidence, merged_confidence, now, winner_id))
            db.execute("""UPDATE memories SET status='duplicate',supersedes_id=?,
              updated_at=? WHERE id=?""", (winner_id, now, int(redundant["id"])))
        elif relation == "supersedes" and result["winner"] in {"left", "right"}:
            current, historical = choose_canonical(left, right, result["winner"])
            winner_id = int(current["id"])
            end_at = int(current["valid_from"] or current["created_at"] or now)
            old_start = int(historical["valid_from"] or 0)
            valid_to = end_at - 1 if end_at > old_start else historical["valid_to"]
            db.execute("""UPDATE memories SET status='historical',valid_to=?,
              supersedes_id=?,updated_at=? WHERE id=?""",
              (valid_to, winner_id, now, int(historical["id"])))
            db.execute("UPDATE memories SET status='active',updated_at=? WHERE id=?",
                       (now, winner_id))
        elif relation == "contradicts":
            group = str(job["pair_key"])
            db.execute("""UPDATE memories SET status='disputed',conflict_group=?,updated_at=?
              WHERE id IN (?,?)""", (group, now, int(left["id"]), int(right["id"])))

    db.execute("""INSERT INTO memory_conflicts
      (left_memory_id,right_memory_id,relation,confidence,reason,winner_id,resolved_at)
      VALUES(?,?,?,?,?,?,?) ON CONFLICT(left_memory_id,right_memory_id) DO UPDATE SET
      relation=excluded.relation,confidence=excluded.confidence,reason=excluded.reason,
      winner_id=excluded.winner_id,resolved_at=excluded.resolved_at""",
      (int(left["id"]), int(right["id"]), relation, confidence,
       result["reason"], winner_id, now))
    db.execute("""UPDATE conflict_jobs SET status='done',relation=?,confidence=?,
      result_json=?,error=NULL,updated_at=? WHERE pair_key=?""",
      (relation, confidence, json.dumps(result, ensure_ascii=False), now, job["pair_key"]))


async def run(args) -> None:
    store = MemoryStore(args.db)
    created = prepare_jobs(store, args.embedding_model, args.top_k, args.min_similarity)
    print(f"候选任务准备完成：新增 {created} 对")
    if args.prepare_only:
        return
    with store.connect() as db:
        db.execute("""UPDATE conflict_jobs SET status='failed',
          error=coalesce(error,'stale running job recovered'),updated_at=?
          WHERE status='running'""", (int(time.time()),))
        jobs = db.execute("""SELECT * FROM conflict_jobs WHERE status IN ('pending','failed')
          ORDER BY left_memory_id,right_memory_id""").fetchall()
    if args.limit:
        jobs = jobs[:args.limit]
    completed = failed = 0
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for start in range(0, len(jobs), args.batch_size):
            batch_jobs = jobs[start:start + args.batch_size]
            pairs = []
            with store.connect() as db:
                for job in batch_jobs:
                    left = db.execute("SELECT * FROM memories WHERE id=?",
                                      (job["left_memory_id"],)).fetchone()
                    right = db.execute("SELECT * FROM memories WHERE id=?",
                                       (job["right_memory_id"],)).fetchone()
                    if left and right:
                        pairs.append((job, left, right))
                db.executemany("""UPDATE conflict_jobs SET status='running',attempts=attempts+1,
                  updated_at=? WHERE pair_key=?""",
                  ((int(time.time()), job["pair_key"]) for job, _, _ in pairs))
            if not pairs:
                continue
            try:
                parse_error = None
                for _ in range(args.json_retries + 1):
                    try:
                        results = await judge_batch(client, args, pairs)
                        parse_error = None
                        break
                    except (ValueError, json.JSONDecodeError) as exc:
                        parse_error = exc
                if parse_error is not None:
                    raise parse_error
                by_index = {result["pair_index"]: result for result in results}
                with store.connect() as db:
                    for pair_index, (job, left, right) in enumerate(pairs):
                        result = by_index.get(pair_index)
                        if result is None:
                            db.execute("""UPDATE conflict_jobs SET status='failed',error=?,updated_at=?
                              WHERE pair_key=?""", ("模型批量响应缺少该 pair_key",
                              int(time.time()), job["pair_key"]))
                            failed += 1
                            continue
                        apply_result(db, job, left, right, result, args.apply_threshold)
                        completed += 1
                print(f"[{min(start + len(pairs), len(jobs))}/{len(jobs)}] "
                      f"批量完成 {len(pairs)} 对")
            except Exception as exc:
                failed += len(pairs)
                with store.connect() as db:
                    db.executemany("""UPDATE conflict_jobs SET status='failed',error=?,updated_at=?
                      WHERE pair_key=?""", ((f"{type(exc).__name__}: {exc}"[:1000],
                      int(time.time()), job["pair_key"]) for job, _, _ in pairs))
                print(f"[{min(start + len(pairs), len(jobs))}/{len(jobs)}] "
                      f"批量失败: {type(exc).__name__}: {exc}")
    print(f"冲突检测结束：完成 {completed}，失败 {failed}")


def main() -> None:
    load_dotenv(BOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="检测、消解并回写长期记忆冲突")
    parser.add_argument("--db", type=Path, default=BOT_DIR / "data" / "memory.db")
    parser.add_argument("--api-base", default=os.getenv("LLM_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen3:8b"))
    parser.add_argument("--embedding-model", default=os.getenv(
        "EMBEDDING_MODEL", "qwen3-embedding:0.6b"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=.72)
    parser.add_argument("--apply-threshold", type=float, default=.82)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--json-retries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--prepare-only", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
