import sys

from src.utils.logger import setup_global_logging
from src.vector_db.backup_manager import backup_chromadb

if __name__ == "__main__":
    setup_global_logging()
    success = backup_chromadb()
    sys.exit(0 if success else 1)
