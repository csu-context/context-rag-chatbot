from contextlib import asynccontextmanager, suppress

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse

from src.utils.paths import BACKUP_DIR
from src.vector_db.backup_manager import backup_chromadb, restore_chromadb


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 기존 백업 상태 파일이 있다면 안전하게 삭제하여 초기화
    status_file = BACKUP_DIR / "backup_status.json"
    if status_file.exists():
        with suppress(Exception):
            status_file.unlink(missing_ok=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/admin/backups/status")
async def get_backup_status():
    """
    현재 진행 중인 백업/복원 작업의 상태를 조회합니다.
    """
    import json
    status_file = BACKUP_DIR / "backup_status.json"
    if not status_file.exists():
        return JSONResponse(content={"status": "idle"})
    try:
        data = json.loads(status_file.read_text(encoding="utf-8"))
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"status": "failed", "error": f"Failed to read status file: {e}"}, status_code=500)


@app.get("/api/admin/backups")
async def get_backups():
    """
    보관 중인 백업 목록을 조회합니다.
    """
    backups = sorted(BACKUP_DIR.glob("chromadb_backup_*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    backup_files = [{"filename": f.name, "created_at": f.stat().st_mtime} for f in backups]
    return JSONResponse(content={"backups": backup_files})


@app.post("/api/admin/backups")
async def create_backup(background_tasks: BackgroundTasks):
    """
    비동기 수동 백업을 강제 실행합니다.
    """
    background_tasks.add_task(backup_chromadb)
    return JSONResponse(content={"message": "Backup job started in the background."})


@app.post("/api/admin/backups/restore")
async def restore_backup(filename: str, background_tasks: BackgroundTasks):
    """
    특정 백업본을 복원하고 자가 진단을 트리거합니다.
    """
    background_tasks.add_task(restore_chromadb, filename)
    return JSONResponse(content={"message": f"Restore job for {filename} started in the background."})
