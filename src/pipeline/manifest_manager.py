import json
import logging
import threading
from pathlib import Path
from typing import Any

from src.utils.file_utils import generate_file_hash
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils.unicode import normalize_path_to_nfc, normalize_to_nfc

logger = logging.getLogger(__name__)


class ManifestManager:
    """manifest.json을 통한 파일 상태 관리 및 변경점(Delta) 계산을 담당합니다."""

    _manifest_lock = threading.Lock()  # manifest 파일 race condition 방지

    def __init__(self, parser_type: str):
        self.parser_type = parser_type
        self.manifest_path = PROCESSED_DATA_DIR / "manifest.json"

    def update_parser_type(self, parser_type: str) -> None:
        self.parser_type = parser_type

    def make_manifest(self, files: dict[str, Any]) -> dict[str, Any]:
        return {"version": "2.0", "global_parser_type": self.parser_type, "files": files}

    def load_manifest(self) -> dict[str, Any]:
        """manifest.json 파일을 로드합니다. 파일이 없거나 손상된 경우 빈 2.0 매니페스트를 반환합니다."""
        default_manifest = self.make_manifest({})
        if not self.manifest_path.exists():
            logger.warning("Manifest 파일이 없어 전체 재색인을 수행합니다.")
            return default_manifest
        try:
            with ManifestManager._manifest_lock, open(self.manifest_path, encoding="utf-8") as f:
                data = json.load(f)

            # 하위 호환성 처리 (1.0 규격)
            if not isinstance(data, dict) or "version" not in data:
                logger.info("이전 규격(1.0)의 Manifest가 감지되어 2.0 규격으로 자동 전환합니다.")
                files_converted = {}
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str):
                            files_converted[k] = {"hash": v, "parser_type": self.parser_type}
                data = self.make_manifest(files_converted)
            return data
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Manifest 파일이 손상되었거나 찾을 수 없어 전체 재색인을 수행합니다.")
            with ManifestManager._manifest_lock:
                if self.manifest_path.exists():
                    self.manifest_path.unlink()
            return default_manifest

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        """처리 완료 후 새로운 manifest 상태를 저장합니다.

        atomic write(임시 파일 → rename)로 race condition 방지.
        """
        tmp_path = self.manifest_path.with_suffix(".tmp")
        try:
            with ManifestManager._manifest_lock:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
                tmp_path.replace(self.manifest_path)  # atomic on POSIX/Windows
            logger.info(f"Manifest 업데이트 완료: {self.manifest_path}")
        except Exception as e:
            logger.error(f"Manifest 저장 실패: {e}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def calculate_delta(
        self,
        all_files: list[Path],
        old_manifest: dict[str, Any],
        storage_manager: Any,
        db_manager: Any,
    ) -> tuple[list[Path], list[str], dict[str, Any]]:
        """현재 파일 상태와 이전 상태를 비교하여 변경 사항(Delta)을 계산합니다."""
        old_files = old_manifest.get("files", {})
        old_files_nfc = {normalize_to_nfc(k): v for k, v in old_files.items()}

        new_files = {}
        for f in all_files:
            rel_path = normalize_path_to_nfc(f.relative_to(RAW_DATA_DIR))
            old_info = old_files.get(rel_path) or old_files_nfc.get(rel_path, {})
            file_parser_type = old_info.get("parser_type", self.parser_type)
            h = generate_file_hash(f, file_parser_type)
            new_files[rel_path] = {"hash": h, "parser_type": file_parser_type}

        new_manifest = self.make_manifest(new_files)

        files_to_process = []
        for rel_path, info in new_files.items():
            old_info = old_files.get(rel_path)
            json_path = storage_manager.get_processed_path(info["hash"])
            db_count = db_manager.get_source_count(Path(rel_path).name)
            if not old_info or old_info.get("hash") != info["hash"] or not json_path.exists() or db_count == 0:
                files_to_process.append(RAW_DATA_DIR / rel_path)

        source_ids_to_delete = []
        for rel_path, old_info in old_files.items():
            new_info = new_files.get(rel_path)
            if not new_info or old_info.get("hash") != new_info["hash"]:
                old_hash = old_info.get("hash")
                if old_hash:
                    source_ids_to_delete.append(old_hash)

        return files_to_process, source_ids_to_delete, new_manifest

    def find_relative_path(self, file_name: str, old_files: dict[str, Any], scanned_files: list[Path]) -> str | None:
        """주어진 파일 이름에 해당하는 상대 경로를 찾습니다."""
        normalized_file_name = normalize_to_nfc(file_name)

        for rel_path in old_files:
            normalized_rel_path = normalize_to_nfc(rel_path)
            if Path(normalized_rel_path).name == normalized_file_name or normalized_rel_path == normalized_file_name:
                return rel_path

        for f in scanned_files:
            if normalize_to_nfc(f.name) == normalized_file_name:
                return normalize_path_to_nfc(f.relative_to(RAW_DATA_DIR))

        return None
