from __future__ import annotations

import tempfile
import time
import unittest
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.memory_store import MemoryStore, format_memory_time, memory_embedding_text
from scripts.resolve_memory_conflicts import apply_result


class MemorySystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_schema_migrates_and_persists_interaction(self):
        self.store.touch_interaction("peer", user_at=100, bot_at=120)
        state = self.store.get_interaction_state("peer")
        self.assertEqual(state["last_user_at"], 100)
        self.assertEqual(state["last_bot_at"], 120)
        with self.store.connect() as db:
            columns = {row["name"] for row in db.execute("pragma table_info(memories)")}
        self.assertIn("supersedes_id", columns)
        self.assertIn("conflict_group", columns)

    def test_embedding_roundtrip_and_hybrid_retrieval(self):
        now = int(time.time())
        self.store.save_memories([
            {"kind": "person", "subject_id": "peer", "summary": "对方喜欢阅读科幻小说",
             "keywords": ["阅读", "科幻", "小说"], "confidence": .9,
             "valid_from": now - 86400 * 10, "valid_to": None},
            {"kind": "person", "subject_id": "peer", "summary": "对方喜欢打篮球",
             "keywords": ["运动", "篮球"], "confidence": .8,
             "valid_from": now - 86400 * 20, "valid_to": None},
        ], ["1"])
        rows = self.store.memories_needing_embeddings("test")
        vectors = {"对方喜欢阅读科幻小说": [1.0, 0.0], "对方喜欢打篮球": [0.0, 1.0]}
        self.store.save_embeddings("test", [
            (row["id"], memory_embedding_text(row), vectors[row["summary"]])
            for row in rows
        ])
        context = self.store.search_context(
            "最近读什么", "self", "peer", limit=1, stable_limit=0,
            query_embedding=[1.0, 0.0], embedding_model="test",
        )
        self.assertIn("科幻小说", context)
        self.assertIn("发生", context.replace("自 ", "发生"))

    def test_runtime_messages_and_extraction_cursor(self):
        self.assertEqual(self.store.save_runtime_message(
            "m1", "peer", "peer", 100, "你好", 1), 1)
        self.assertEqual(self.store.save_runtime_message(
            "m1", "peer", "peer", 100, "你好", 1), 0)
        batch = self.store.get_online_extraction_batch("peer")
        self.assertEqual(len(batch), 1)
        self.store.finish_online_extraction("peer", batch[-1]["id"])
        self.assertEqual(self.store.get_online_extraction_batch("peer"), [])

    def test_readable_time(self):
        label = format_memory_time("episode", 1_700_000_000, None, 1_700_086_400)
        self.assertIn("发生于", label)
        self.assertIn("1天前", label)

    def test_conflict_supersedes_writes_history(self):
        self.store.save_memories([
            {"kind": "person", "subject_id": "peer", "summary": "对方住在北京",
             "keywords": ["居住地", "北京"], "confidence": .85,
             "valid_from": 100, "valid_to": None},
            {"kind": "person", "subject_id": "peer", "summary": "对方已经搬到上海",
             "keywords": ["居住地", "上海"], "confidence": .9,
             "valid_from": 200, "valid_to": None},
        ], ["m1"])
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM memories ORDER BY valid_from").fetchall()
            db.execute("""INSERT INTO conflict_jobs
              (pair_key,left_memory_id,right_memory_id,status,updated_at)
              VALUES('pair',?,?, 'running',1)""", (rows[0]["id"], rows[1]["id"]))
            job = db.execute("SELECT * FROM conflict_jobs WHERE pair_key='pair'").fetchone()
            apply_result(db, job, rows[0], rows[1], {
                "relation": "supersedes", "winner": "right",
                "confidence": .95, "reason": "搬家形成新状态",
            }, .82)
        with self.store.connect() as db:
            old, new = db.execute("SELECT * FROM memories ORDER BY valid_from").fetchall()
            self.assertEqual(old["status"], "historical")
            self.assertEqual(old["valid_to"], 199)
            self.assertEqual(old["supersedes_id"], new["id"])
            self.assertEqual(new["status"], "active")


if __name__ == "__main__":
    unittest.main()
