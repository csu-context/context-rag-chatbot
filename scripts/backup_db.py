import logging
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가하여 src 패키지 임포트 가능하게 설정
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.paths import BACKUP_DIR, LOGS_DIR, VECTOR_DB_DIR, ensure_directories  # noqa: E402

# 로깅 설정
LOG_FILE = LOGS_DIR / "backup.log"
ensure_directories()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def backup_chromadb(rotation_limit=5):
    """
    ChromaDB의 vector_db 디렉토리를 압축하여 백업합니다.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"chromadb_backup_{timestamp}.tar.gz"
    backup_path = BACKUP_DIR / backup_filename

    logger.info(f"백업 시작: {VECTOR_DB_DIR} -> {backup_path}")

    if not VECTOR_DB_DIR.exists():
        logger.error(f"백업 실패: 소스 디렉토리 {VECTOR_DB_DIR}가 존재하지 않습니다.")
        return False

    try:
        # data/backups 디렉토리 확인
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        # 압축 실행
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(VECTOR_DB_DIR, arcname=VECTOR_DB_DIR.name)

        logger.info(f"백업 완료: {backup_filename}")

        # 순환(Rotation) 정책 적용
        rotate_backups(rotation_limit)
        return True
    except Exception as e:
        logger.exception(f"백업 중 오류 발생: {e}")
        return False


def rotate_backups(limit):
    """
    오래된 백업 파일을 삭제하여 개수를 유지합니다.
    """
    try:
        backups = sorted(list(BACKUP_DIR.glob("chromadb_backup_*.tar.gz")), key=lambda x: x.name)

        if len(backups) > limit:
            to_delete = backups[:-limit]
            for file in to_delete:
                file.unlink()
                logger.info(f"오래된 백업 삭제됨: {file.name}")
    except Exception as e:
        logger.error(f"백업 순환 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    success = backup_chromadb()
    sys.exit(0 if success else 1)
