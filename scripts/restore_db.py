import argparse
import sys

from src.utils.logger import setup_global_logging
from src.vector_db.backup_manager import restore_chromadb

if __name__ == "__main__":
    setup_global_logging()
    parser = argparse.ArgumentParser(description="Restore ChromaDB from a backup archive.")
    parser.add_argument("--file", help="Backup filename to restore. Defaults to the newest backup.")
    args = parser.parse_args()

    success = restore_chromadb(args.file)
    sys.exit(0 if success else 1)
