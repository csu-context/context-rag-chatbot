from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse

from src.utils.paths import BACKUP_DIR
from src.vector_db.backup_manager import backup_chromadb, restore_chromadb

app = FastAPI()


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
