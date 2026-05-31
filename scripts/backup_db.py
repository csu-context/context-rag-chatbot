# ruff: noqa: E402, I001
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.backup_manager import backup_chromadb, rotate_backups as rotate_backups
from src.utils.logger import setup_global_logging


if __name__ == "__main__":
    setup_global_logging()
    success = backup_chromadb()
    sys.exit(0 if success else 1)
