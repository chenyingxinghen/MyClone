"""为已有记忆批量生成向量；可重复运行，只处理缺失或内容变化的条目。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOT_DIR = Path(__file__).resolve().parents[1]

from components.embedding_client import OllamaEmbeddingClient
from components.memory_store import MemoryStore, memory_embedding_text


def main() -> None:
    load_dotenv(BOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="使用 Ollama 为记忆批量回填 embedding")
    parser.add_argument("--db", type=Path, default=BOT_DIR / "data" / "memory.db")
    parser.add_argument("--api-base", default=os.getenv(
        "EMBEDDING_API_BASE", "http://127.0.0.1:11434"))
    parser.add_argument("--api-key", default=os.getenv(
        "EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "")))
    parser.add_argument("--model", default=os.getenv(
        "EMBEDDING_MODEL", "qwen3-embedding:0.6b"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--num-gpu", type=int, default=0,
                        help="embedding 使用的 GPU 层数；默认 0，即完全在 CPU 上运行")
    parser.add_argument("--keep-alive", default="30m")
    args = parser.parse_args()

    store = MemoryStore(args.db)
    pending = store.memories_needing_embeddings(args.model, args.limit)
    if not pending:
        print(f"无需回填：模型 {args.model} 的向量均为最新")
        return

    client = OllamaEmbeddingClient(args.api_base, args.model, args.timeout,
                                   args.api_key, args.num_gpu, args.keep_alive)
    saved = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        texts = [memory_embedding_text(row) for row in batch]
        embeddings = client.embed(texts)
        saved += store.save_embeddings(args.model, [
            (int(row["id"]), text, embedding)
            for row, text, embedding in zip(batch, texts, embeddings)
        ])
        print(f"已写入 {saved}/{len(pending)}")
    print(f"回填完成：模型 {args.model}，共写入 {saved} 条")


if __name__ == "__main__":
    main()
