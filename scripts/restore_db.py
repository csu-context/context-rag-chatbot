import argparse
import logging
import shutil
import sys
import tarfile
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가하여 src 패키지 임포트 가능하게 설정
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.paths import BACKUP_DIR, LOGS_DIR, VECTOR_DB_DIR, ensure_directories  # noqa: E402
from src.vector_db.chroma_manager import ChromaDBManager  # noqa: E402

# 로깅 설정
LOG_FILE = LOGS_DIR / "restore.log"
ensure_directories()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def restore_chromadb(backup_file=None):
    """
    백업 파일을 사용하여 ChromaDB를 복원합니다.
    """
    if backup_file:
        backup_path = BACKUP_DIR / backup_file
    else:
        # 최신 백업 파일 찾기
        backups = sorted(list(BACKUP_DIR.glob("chromadb_backup_*.tar.gz")), key=lambda x: x.name, reverse=True)
        if not backups:
            logger.error("복원 실패: 백업 파일이 data/backups/ 경로에 존재하지 않습니다.")
            return False
        backup_path = backups[0]

    if not backup_path.exists():
        logger.error(f"복원 실패: 백업 파일 {backup_path}를 찾을 수 없습니다.")
        return False

    logger.info(f"복원 프로세스 시작: {backup_path.name}")

    # 1. 기존 vector_db 제거 (Lock 방지 및 클린 복원)
    try:
        if VECTOR_DB_DIR.exists():
            logger.info(f"기존 데이터 삭제 중: {VECTOR_DB_DIR}")
            shutil.rmtree(VECTOR_DB_DIR)

        # 2. 압축 해제
        logger.info("백업 압축 해제 중...")
        with tarfile.open(backup_path, "r:gz") as tar:
            # arcname=VECTOR_DB_DIR.name 으로 압축했으므로,
            # 해제 시 프로젝트 루트(BASE_DIR)에서 해제하면 vector_db/ 가 생성됨
            tar.extractall(path=BASE_DIR)

        logger.info("복원 완료. DB 상태 자가 진단을 시작합니다.")

        # 3. 자가 진단
        return diagnose_db()

    except PermissionError:
        logger.error(
            "복원 실패: 파일 잠금(Lock)이 발생했습니다. ChromaDB를 사용하는 다른 프로세스를 종료 후 다시 시도하십시오."
        )
        return False
    except Exception as e:
        logger.exception(f"복원 중 오류 발생: {e}")
        return False


def diagnose_db():
    """
    복원된 DB의 상태를 점검합니다.
    """
    try:
        # ChromaDBManager 초기화 시도
        manager = ChromaDBManager()
        count = manager.get_count()
        logger.info(f"자가 진단 성공: ChromaDB 연결 확인됨. 현재 데이터 개수: {count}")
        return True
    except Exception as e:
        logger.error(f"자가 진단 실패: DB 연결 또는 데이터 조회 중 오류 발생: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB 복원 스크립트")
    parser.add_argument("--file", help="복원할 백업 파일명 (기본값: 가장 최근 백업)")
    args = parser.parse_args()

    success = restore_chromadb(args.file)
    sys.exit(0 if success else 1)
