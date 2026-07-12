from __future__ import annotations

import json
import hashlib
import math
import re
import sqlite3
import time
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, msg_id TEXT UNIQUE, sender_id TEXT NOT NULL,
  peer_id TEXT NOT NULL, sent_at INTEGER NOT NULL, text TEXT NOT NULL,
  send_type INTEGER, source_file TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_people_time
  ON messages(sender_id, peer_id, sent_at);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  text, content='messages', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid,text) VALUES(new.id,new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,text) VALUES('delete',old.id,old.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts,rowid,text) VALUES('delete',old.id,old.text);
  INSERT INTO messages_fts(rowid,text) VALUES(new.id,new.text);
END;
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, subject_id TEXT NOT NULL,
  object_id TEXT, summary TEXT NOT NULL, keywords TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT .7, valid_from INTEGER, valid_to INTEGER,
  status TEXT NOT NULL DEFAULT 'active', evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
  fingerprint TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(subject_id,status,kind);
CREATE TABLE IF NOT EXISTS memory_embeddings (
  memory_id INTEGER NOT NULL, model TEXT NOT NULL, dimensions INTEGER NOT NULL,
  embedding BLOB NOT NULL, content_hash TEXT NOT NULL, embedded_at INTEGER NOT NULL,
  PRIMARY KEY(memory_id,model), FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model ON memory_embeddings(model,memory_id);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  summary, keywords, content='memories', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid,summary,keywords) VALUES(new.id,new.summary,new.keywords);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts,rowid,summary,keywords)
    VALUES('delete',old.id,old.summary,old.keywords);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts,rowid,summary,keywords)
    VALUES('delete',old.id,old.summary,old.keywords);
  INSERT INTO memories_fts(rowid,summary,keywords) VALUES(new.id,new.summary,new.keywords);
END;
CREATE TABLE IF NOT EXISTS extraction_jobs (
  chunk_key TEXT PRIMARY KEY, start_time INTEGER NOT NULL, end_time INTEGER NOT NULL,
  message_ids TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0, error TEXT, updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS prepared_conversations (
  chunk_key TEXT PRIMARY KEY, peer_id TEXT NOT NULL,
  start_time INTEGER NOT NULL, end_time INTEGER NOT NULL,
  message_ids TEXT NOT NULL, message_count INTEGER NOT NULL,
  informative_count INTEGER NOT NULL, speaker_count INTEGER NOT NULL,
  quality_score REAL NOT NULL, keep INTEGER NOT NULL,
  reason TEXT NOT NULL, prepared_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prepared_peer_keep
  ON prepared_conversations(peer_id,keep,start_time);
CREATE TABLE IF NOT EXISTS preprocessing_runs (
  id INTEGER PRIMARY KEY, peer_id TEXT NOT NULL, run_at INTEGER NOT NULL,
  source_messages INTEGER NOT NULL, removed_placeholders INTEGER NOT NULL,
  prepared_chunks INTEGER NOT NULL, kept_chunks INTEGER NOT NULL,
  rules_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS interaction_state (
  peer_id TEXT PRIMARY KEY, last_user_at INTEGER, last_bot_at INTEGER,
  updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS online_extraction_state (
  peer_id TEXT PRIMARY KEY, last_message_rowid INTEGER NOT NULL DEFAULT 0,
  last_extracted_at INTEGER, status TEXT NOT NULL DEFAULT 'idle',
  error TEXT, updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS conflict_jobs (
  pair_key TEXT PRIMARY KEY, left_memory_id INTEGER NOT NULL,
  right_memory_id INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0, relation TEXT, confidence REAL,
  result_json TEXT, error TEXT, updated_at INTEGER NOT NULL,
  UNIQUE(left_memory_id,right_memory_id)
);
CREATE TABLE IF NOT EXISTS memory_conflicts (
  id INTEGER PRIMARY KEY, left_memory_id INTEGER NOT NULL,
  right_memory_id INTEGER NOT NULL, relation TEXT NOT NULL,
  confidence REAL NOT NULL, reason TEXT NOT NULL DEFAULT '',
  winner_id INTEGER, resolved_at INTEGER NOT NULL,
  UNIQUE(left_memory_id,right_memory_id)
);
"""


MEMORY_EXTRA_COLUMNS = {
    "supersedes_id": "INTEGER",
    "conflict_group": "TEXT",
}


def memory_embedding_text(row: sqlite3.Row | dict) -> str:
    kind = str(row["kind"])
    summary = str(row["summary"])
    keywords = str(row["keywords"] or "")
    return f"记忆类型：{kind}\n内容：{summary}\n关键词：{keywords}".strip()


def embedding_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_embedding(values: list[float]) -> tuple[bytes, int]:
    vector = array("f", (float(x) for x in values))
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if not vector or norm <= 0:
        raise ValueError("embedding 不能为空或零向量")
    normalized = array("f", (float(x) / norm for x in vector))
    return normalized.tobytes(), len(normalized)


def unpack_embedding(blob: bytes, dimensions: int) -> array:
    vector = array("f")
    vector.frombytes(blob)
    if len(vector) != dimensions:
        raise ValueError(f"embedding 维度不匹配：期望 {dimensions}，实际 {len(vector)}")
    return vector


def _relative_time(timestamp: int, now: int) -> str:
    delta = now - timestamp
    future = delta < 0
    seconds = abs(delta)
    if seconds < 3600:
        amount, unit = max(1, int(seconds // 60)), "分钟"
    elif seconds < 86400:
        amount, unit = int(seconds // 3600), "小时"
    elif seconds < 86400 * 30:
        amount, unit = int(seconds // 86400), "天"
    elif seconds < 86400 * 365:
        amount, unit = int(seconds // (86400 * 30)), "个月"
    else:
        amount, unit = int(seconds // (86400 * 365)), "年"
    return f"约{amount}{unit}{'后' if future else '前'}"


def format_memory_time(kind: str, valid_from: int | None, valid_to: int | None,
                       now: int | None = None) -> str:
    now = int(time.time()) if now is None else now
    if valid_from is None and valid_to is None:
        return "时间未知"
    start = datetime.fromtimestamp(valid_from).strftime("%Y-%m-%d") if valid_from else None
    end = datetime.fromtimestamp(valid_to).strftime("%Y-%m-%d") if valid_to else None
    if start and end:
        return f"有效期 {start} 至 {end}"
    if end:
        return f"截至 {end}"
    relative = _relative_time(int(valid_from), now)
    if kind == "episode":
        return f"发生于 {start}，{relative}"
    return f"自 {start} 起，{relative}记录"


def _terms(text: str) -> list[str]:
    """提取查询词；保留中文短词，避免 trigram 对 2 字词完全失配。"""
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]{2,}", text.lower())
    terms: list[str] = []
    for chunk in chunks:
        terms.append(chunk)
        # 自然聊天通常没有空格。补充 2~4 字窗口后，“你还喜欢猫吗”也能命中“喜欢/猫咪”。
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk) and len(chunk) > 4:
            for size in (2, 3, 4):
                terms.extend(chunk[i:i + size] for i in range(len(chunk) - size + 1))
    return list(dict.fromkeys(terms))[:36]


def _ngrams(text: str, size: int = 2) -> set[str]:
    compact = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]", text.lower()))
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[i:i + size] for i in range(len(compact) - size + 1)}


def _relevance(row: sqlite3.Row, terms: list[str], query_grams: set[str]) -> float:
    summary = row["summary"].lower()
    keywords = row["keywords"].lower()
    keyword_tokens = set(keywords.split())
    score = 0.0
    matched_terms = 0
    for term in terms:
        matched = False
        if term in keyword_tokens:
            score += 12.0
            matched = True
        elif term in keywords:
            score += 7.0
            matched = True
        if term in summary:
            score += 4.0
            matched = True
        matched_terms += int(matched)
    if terms:
        score += 6.0 * matched_terms / len(terms)
    overlap = query_grams & _ngrams(f"{keywords} {summary}")
    score += min(len(overlap), 12) * 0.8
    # 相关度主导排序，置信度只负责打破接近的候选，避免高置信无关记忆挤占位置。
    score += float(row["confidence"]) * 1.5
    return score


def _has_temporal_intent(query: str) -> bool:
    return bool(re.search(
        r"今天|昨天|前天|明天|最近|现在|目前|当时|以前|后来|上次|多久|哪天|"
        r"今年|去年|本周|上周|这几天|过去|曾经|\d{4}年|\d{1,2}月",
        query,
    ))


def _dot(left: array, right: array) -> float:
    if len(left) != len(right):
        return -1.0
    return sum(a * b for a, b in zip(left, right))


@dataclass
class MemoryStore:
    path: Path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
            for name, declaration in MEMORY_EXTRA_COLUMNS.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def import_history(self, source: Path, peer_id: str) -> int:
        records = json.loads(source.read_text(encoding="utf-8-sig"))
        rows = []
        for item in records:
            text = "".join(str(x) for x in (item.get("elements") or [])).strip()
            sender = str(item.get("senderQQNum") or "0")
            if not text or sender == "0":
                continue
            rows.append((str(item.get("msgId")), sender, peer_id,
                         int(item.get("msgTime") or 0), text,
                         int(item.get("sendType") or 0), str(source)))
        with self.connect() as db:
            before = db.execute("SELECT count(*) FROM messages").fetchone()[0]
            db.executemany("""INSERT OR IGNORE INTO messages
              (msg_id,sender_id,peer_id,sent_at,text,send_type,source_file)
              VALUES(?,?,?,?,?,?,?)""", rows)
            after = db.execute("SELECT count(*) FROM messages").fetchone()[0]
            return after - before

    def search_context(self, query: str, self_id: str, partner_id: str,
                       limit: int = 24, stable_limit: int = 6,
                       max_chars: int = 5000,
                       query_embedding: list[float] | None = None,
                       embedding_model: str | None = None,
                       vector_min_similarity: float = .25) -> str:
        terms = _terms(query)
        query_grams = _ngrams(query)
        subjects = (self_id, partner_id, f"relationship:{self_id}:{partner_id}")
        with self.connect() as db:
            rows = db.execute("""SELECT id,kind,subject_id,summary,keywords,confidence,
              valid_from,valid_to,status,updated_at,conflict_group FROM memories
              WHERE status IN ('active','disputed') AND subject_id IN (?,?,?)
              AND (valid_to IS NULL OR valid_to>=strftime('%s','now'))""",
              subjects).fetchall()
            embedding_rows = []
            if query_embedding and embedding_model:
                embedding_rows = db.execute("""SELECT e.memory_id,e.dimensions,e.embedding
                  FROM memory_embeddings e JOIN memories m ON m.id=e.memory_id
                  WHERE e.model=? AND m.status IN ('active','disputed')
                  AND m.subject_id IN (?,?,?)
                  AND (m.valid_to IS NULL OR m.valid_to>=strftime('%s','now'))""",
                  (embedding_model, *subjects)).fetchall()
        if not rows:
            return ""

        by_id = {int(row["id"]): row for row in rows}
        lexical = [(_relevance(row, terms, query_grams), row) for row in rows]
        lexical = [item for item in lexical
                   if item[0] > float(item[1]["confidence"]) * 1.5]
        lexical.sort(key=lambda item: item[0], reverse=True)
        lexical_rank = {int(row["id"]): rank
                        for rank, (_, row) in enumerate(lexical[:60], 1)}

        vector_scores: list[tuple[float, int]] = []
        if embedding_rows and query_embedding:
            packed_query, dimensions = pack_embedding(query_embedding)
            query_vector = unpack_embedding(packed_query, dimensions)
            for embedded in embedding_rows:
                try:
                    vector = unpack_embedding(embedded["embedding"], embedded["dimensions"])
                except ValueError:
                    continue
                similarity = _dot(query_vector, vector)
                if similarity >= vector_min_similarity:
                    vector_scores.append((similarity, int(embedded["memory_id"])))
            vector_scores.sort(reverse=True)
        vector_rank = {memory_id: rank
                       for rank, (_, memory_id) in enumerate(vector_scores[:60], 1)}

        # Reciprocal Rank Fusion：词面和向量分数量纲不同，用排名融合更稳定。
        candidate_ids = set(lexical_rank) | set(vector_rank)
        temporal_intent = _has_temporal_intent(query)
        recent = sorted(
            (by_id[mid] for mid in candidate_ids if by_id[mid]["valid_from"]),
            key=lambda row: row["valid_from"], reverse=True,
        )
        temporal_rank = {int(row["id"]): rank for rank, row in enumerate(recent, 1)}

        def fused(memory_id: int) -> float:
            score = 0.0
            if memory_id in vector_rank:
                score += 1.0 / (60 + vector_rank[memory_id])
            if memory_id in lexical_rank:
                score += .85 / (60 + lexical_rank[memory_id])
            row = by_id[memory_id]
            if memory_id in temporal_rank and (temporal_intent or row["kind"] == "episode"):
                weight = .35 if temporal_intent else .10
                score += weight / (60 + temporal_rank[memory_id])
            score += .05 * float(row["confidence"]) / 61
            return score

        selected_related = sorted(
            (by_id[mid] for mid in candidate_ids),
            key=lambda row: (fused(int(row["id"])), row["confidence"], row["updated_at"]),
            reverse=True,
        )[:limit]
        used = {row["summary"] for row in selected_related}
        stable = sorted(
            (row for row in rows if row["summary"] not in used and row["kind"] != "episode"),
            key=lambda row: (row["confidence"], row["updated_at"]), reverse=True,
        )[:stable_limit]

        sections = []
        if selected_related:
            sections.append(("【当前话题相关】", selected_related))
        if stable:
            sections.append(("【稳定背景】", stable))
        lines: list[str] = []
        length = 0
        for heading, section_rows in sections:
            if lines:
                lines.append("")
                length += 1
            lines.append(heading)
            length += len(heading)
            for row in section_rows:
                time_label = format_memory_time(row["kind"], row["valid_from"], row["valid_to"])
                conflict_label = "，信息存在未决冲突" if row["status"] == "disputed" else ""
                line = (f"- [{row['kind']}，{time_label}{conflict_label}] {row['summary']}"
                        f"（可信度 {row['confidence']:.2f}）")
                if length + len(line) + 1 > max_chars:
                    return "\n".join(lines)
                lines.append(line)
                length += len(line) + 1
        return "\n".join(lines)

    def memories_needing_embeddings(self, model: str, limit: int = 0) -> list[sqlite3.Row]:
        # The hash depends on each row, so stale rows are filtered in Python.
        with self.connect() as db:
            rows = db.execute("""SELECT m.id,m.kind,m.summary,m.keywords,e.content_hash
              FROM memories m LEFT JOIN memory_embeddings e
              ON e.memory_id=m.id AND e.model=? ORDER BY m.id""", (model,)).fetchall()
        pending = [row for row in rows
                   if row["content_hash"] != embedding_content_hash(memory_embedding_text(row))]
        return pending[:limit] if limit else pending

    def save_embeddings(self, model: str,
                        rows: list[tuple[int, str, list[float]]]) -> int:
        now = int(time.time())
        packed = []
        for memory_id, text, values in rows:
            blob, dimensions = pack_embedding(values)
            packed.append((memory_id, model, dimensions, blob,
                           embedding_content_hash(text), now))
        with self.connect() as db:
            db.executemany("""INSERT INTO memory_embeddings
              (memory_id,model,dimensions,embedding,content_hash,embedded_at)
              VALUES(?,?,?,?,?,?) ON CONFLICT(memory_id,model) DO UPDATE SET
              dimensions=excluded.dimensions,embedding=excluded.embedding,
              content_hash=excluded.content_hash,embedded_at=excluded.embedded_at""", packed)
        return len(packed)

    def save_runtime_message(self, msg_id: str, sender_id: str, peer_id: str,
                             sent_at: int, text: str, send_type: int) -> int:
        with self.connect() as db:
            cursor = db.execute("""INSERT OR IGNORE INTO messages
              (msg_id,sender_id,peer_id,sent_at,text,send_type,source_file)
              VALUES(?,?,?,?,?,?,'runtime')""",
              (msg_id, sender_id, peer_id, sent_at, text, send_type))
            return cursor.rowcount

    def get_interaction_state(self, peer_id: str) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute("SELECT * FROM interaction_state WHERE peer_id=?",
                              (peer_id,)).fetchone()

    def touch_interaction(self, peer_id: str, *, user_at: int | None = None,
                          bot_at: int | None = None) -> None:
        now = int(time.time())
        with self.connect() as db:
            db.execute("""INSERT INTO interaction_state
              (peer_id,last_user_at,last_bot_at,updated_at) VALUES(?,?,?,?)
              ON CONFLICT(peer_id) DO UPDATE SET
              last_user_at=coalesce(excluded.last_user_at,last_user_at),
              last_bot_at=coalesce(excluded.last_bot_at,last_bot_at),
              updated_at=excluded.updated_at""", (peer_id, user_at, bot_at, now))

    def get_online_extraction_batch(self, peer_id: str,
                                    limit: int = 40) -> list[sqlite3.Row]:
        with self.connect() as db:
            state = db.execute("SELECT last_message_rowid FROM online_extraction_state "
                               "WHERE peer_id=?", (peer_id,)).fetchone()
            after_id = int(state["last_message_rowid"]) if state else 0
            return db.execute("""SELECT id,msg_id,sender_id,sent_at,text FROM messages
              WHERE peer_id=? AND source_file='runtime' AND id>?
              ORDER BY id LIMIT ?""", (peer_id, after_id, limit)).fetchall()

    def finish_online_extraction(self, peer_id: str, last_message_rowid: int | None,
                                 error: str | None = None) -> None:
        now = int(time.time())
        status = "failed" if error else "idle"
        with self.connect() as db:
            db.execute("""INSERT INTO online_extraction_state
              (peer_id,last_message_rowid,last_extracted_at,status,error,updated_at)
              VALUES(?,?,?,?,?,?) ON CONFLICT(peer_id) DO UPDATE SET
              last_message_rowid=CASE WHEN excluded.last_message_rowid>0
                THEN excluded.last_message_rowid ELSE last_message_rowid END,
              last_extracted_at=CASE WHEN excluded.error IS NULL
                THEN excluded.last_extracted_at ELSE last_extracted_at END,
              status=excluded.status,error=excluded.error,updated_at=excluded.updated_at""",
              (peer_id, last_message_rowid or 0, None if error else now,
               status, error[:1000] if error else None, now))

    def memory_ids_for_items(self, items: list[dict]) -> list[int]:
        fingerprints = []
        for item in items:
            kind = str(item.get("kind", "")).strip()
            subject = str(item.get("subject_id", "")).strip()
            summary = str(item.get("summary", "")).strip()
            if kind and subject and summary:
                fingerprints.append(f"{kind}|{subject}|{summary}")
        if not fingerprints:
            return []
        marks = ",".join("?" for _ in fingerprints)
        with self.connect() as db:
            return [int(row["id"]) for row in db.execute(
                f"SELECT id FROM memories WHERE fingerprint IN ({marks})", fingerprints)]

    def memories_by_ids(self, memory_ids: list[int]) -> list[sqlite3.Row]:
        if not memory_ids:
            return []
        marks = ",".join("?" for _ in memory_ids)
        with self.connect() as db:
            return db.execute(f"SELECT id,kind,summary,keywords FROM memories "
                              f"WHERE id IN ({marks})", memory_ids).fetchall()

    def save_memories(self, items: list[dict], evidence_ids: list[str]) -> int:
        now = int(time.time())
        rows = []
        for x in items:
            summary = str(x.get("summary", "")).strip()
            subject = str(x.get("subject_id", "")).strip()
            kind = str(x.get("kind", "")).strip()
            if not summary or not subject or not kind:
                continue
            fingerprint = f"{kind}|{subject}|{summary}"
            rows.append((kind, subject, x.get("object_id"), summary,
                         " ".join(x.get("keywords") or []),
                         float(x.get("confidence", .7)), x.get("valid_from"),
                         x.get("valid_to"), json.dumps(evidence_ids, ensure_ascii=False),
                         now, now, fingerprint))
        with self.connect() as db:
            before = db.execute("SELECT count(*) FROM memories").fetchone()[0]
            db.executemany("""INSERT INTO memories
              (kind,subject_id,object_id,summary,keywords,confidence,valid_from,valid_to,
               evidence_json,created_at,updated_at,fingerprint)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(fingerprint) DO UPDATE SET
              confidence=max(confidence,excluded.confidence),updated_at=excluded.updated_at,
              evidence_json=excluded.evidence_json""", rows)
            after = db.execute("SELECT count(*) FROM memories").fetchone()[0]
            return after - before
