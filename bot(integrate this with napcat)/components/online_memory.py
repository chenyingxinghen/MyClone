from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from components.embedding_client import OllamaEmbeddingClient, is_local_ollama, ollama_root
from components.memory_extraction import PROMPT, parse_json_array, salvage_json_array
from components.memory_store import MemoryStore, memory_embedding_text


@dataclass
class OnlineMemoryExtractor:
    store: MemoryStore
    self_id: str
    llm_api_base: str
    llm_model: str
    llm_api_key: str = ""
    embedding_client: OllamaEmbeddingClient | None = None
    min_messages: int = 8
    max_messages: int = 40
    max_items: int = 8
    timeout: float = 120.0
    llm_keep_alive: str = "30m"

    async def maybe_extract(self, peer_id: str) -> int:
        rows = self.store.get_online_extraction_batch(peer_id, self.max_messages)
        if len(rows) < self.min_messages:
            return 0
        semantic_chars = sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", row["text"]))
                             for row in rows)
        speakers = {str(row["sender_id"]) for row in rows}
        last_rowid = int(rows[-1]["id"])
        if semantic_chars < 60 or len(speakers) < 2:
            self.store.finish_online_extraction(peer_id, last_rowid)
            return 0

        dialogue = "\n".join(
            f"[{row['sent_at']}] QQ{row['sender_id']}: {row['text']}" for row in rows
        )
        prompt = PROMPT.format(
            self_id=self.self_id,
            partner_id=peer_id,
            max_items=self.max_items,
            dialogue=dialogue,
        )
        local = is_local_ollama(self.llm_api_base)
        root = ollama_root(self.llm_api_base) if local else self.llm_api_base.rstrip("/")
        url = f"{root}/api/chat" if local else f"{root}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if local:
            payload.update({"think": False, "keep_alive": self.llm_keep_alive,
                            "options": {"temperature": .1, "num_predict": 1800}})
        else:
            payload.update({"temperature": .1, "max_tokens": 1800})

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            if local:
                content = data.get("message", {}).get("content", "")
            else:
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                items = parse_json_array(content)
            except Exception:
                items = salvage_json_array(content)
            relation_id = f"relationship:{self.self_id}:{peer_id}"
            normalized = []
            for item in items[:self.max_items]:
                if not isinstance(item, dict) or item.get("kind") not in {
                    "self", "person", "relationship", "episode"
                }:
                    continue
                item["subject_id"] = ({"self": self.self_id, "person": peer_id}
                                      .get(item["kind"], relation_id))
                item["confidence"] = min(.95, max(.05, float(item.get("confidence", .7))))
                if item.get("valid_from") is None:
                    item["valid_from"] = int(rows[-1]["sent_at"])
                normalized.append(item)
            self.store.save_memories(normalized, [str(row["msg_id"]) for row in rows])
            if normalized and self.embedding_client:
                try:
                    memory_rows = self.store.memories_by_ids(
                        self.store.memory_ids_for_items(normalized))
                    texts = [memory_embedding_text(row) for row in memory_rows]
                    embeddings = await self.embedding_client.embed_async(texts)
                    self.store.save_embeddings(self.embedding_client.model, [
                        (int(row["id"]), text, vector)
                        for row, text, vector in zip(memory_rows, texts, embeddings)
                    ])
                except Exception:
                    # 记忆正文已经成功落库；向量可以由回填脚本稍后补齐。
                    pass
            self.store.finish_online_extraction(peer_id, last_rowid)
            return len(normalized)
        except Exception as exc:
            self.store.finish_online_extraction(peer_id, None,
                                                f"{type(exc).__name__}: {exc}")
            raise
