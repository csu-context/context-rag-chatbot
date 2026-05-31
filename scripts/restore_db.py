# ruff: noqa: E402, I001
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.backup_manager import diagnose_db as diagnose_db, restore_chromadb
from src.utils.logger import setup_global_logging


if __name__ == "__main__":
    setup_global_logging()
    parser = argparse.ArgumentParser(description="Restore ChromaDB from a backup archive.")
    parser.add_argument("--file", help="Backup filename to restore. Defaults to the newest backup.")
    args = parser.parse_args()

    success = restore_chromadb(args.file)
    sys.exit(0 if success else 1)
