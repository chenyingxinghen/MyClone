import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOT_DIR = Path(__file__).resolve().parents[1]

from components.memory_store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="导入 QQ 私聊历史到认知记忆库")
    parser.add_argument("history", type=Path)
    parser.add_argument("--peer", required=True, help="交谈对象 QQ")
    parser.add_argument("--db", type=Path, default=BOT_DIR / "data" / "memory.db")
    args = parser.parse_args()
    count = MemoryStore(args.db).import_history(args.history, args.peer)
    print(f"导入完成：新增 {count} 条消息，数据库 {args.db.resolve()}")


if __name__ == "__main__":
    main()
