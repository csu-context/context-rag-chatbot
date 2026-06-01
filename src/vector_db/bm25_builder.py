import hashlib
import json
import logging
import traceback
from pathlib import Path
from typing import Any

from src.common.constants import DataFields, MetadataFields
from src.vector_db.bm25_index import BM25PlusIndex
from src.vector_db.bm25_tokenizer import BM25Tokenizer

logger = logging.getLogger(__name__)

_CORPUS_FILE = "corpus.json"
_MANIFEST_FILE = "manifest.json"


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


class BM25Builder:
    """JSON 파일들을 스캔하고 파싱하여 BM25 코퍼스와 인덱스를 빌드하는 책임을 갖습니다."""

    def __init__(self, data_dir: Path, cache_dir: Path, tokenizer: BM25Tokenizer):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer

    def _flatten_data(self, data: Any) -> list[dict]:
        flattened = []

        if isinstance(data, list):
            for item in data:
                flattened.extend(self._flatten_data(item))
            return flattened

        if isinstance(data, dict):
            text_content = data.get(DataFields.TEXT) or data.get(DataFields.CONTENT) or data.get(DataFields.PARENT_TEXT)

            if text_content:
                node: dict = {
                    DataFields.CONTENT: text_content,
                    DataFields.METADATA: data.get(DataFields.METADATA) or {},
                }
                # chunk_id가 최상위에 존재하면 보존 (RRF 중복 제거용)
                if MetadataFields.CHUNK_ID in data:
                    node[MetadataFields.CHUNK_ID] = data[MetadataFields.CHUNK_ID]
                flattened.append(node)

            children = data.get(DataFields.CHILDREN)
            if children and isinstance(children, list):
                for child in children:
                    flattened.extend(self._flatten_data(child))

        return flattened

    def _get_all_json_files(self) -> list[Path]:
        return [f for f in self.data_dir.glob("*.json") if f.name != "manifest.json"]

    def _load_existing_cache(self) -> tuple[list, dict, BM25PlusIndex | None]:
        """기존 캐시 코퍼스와 매니페스트를 로드합니다. 실패 시 빈 값 반환."""
        corpus_path = self.cache_dir / _CORPUS_FILE
        manifest_path = self.cache_dir / _MANIFEST_FILE
        if not (corpus_path.exists() and manifest_path.exists()):
            return [], {}, None
        try:
            with open(corpus_path, encoding="utf-8") as f:
                corpus = json.load(f)
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            bm25 = BM25PlusIndex.load(self.cache_dir)
            return corpus, manifest, bm25
        except Exception as cache_err:
            logger.warning(f"기존 캐시 로드 실패 (전체 재구성): {cache_err}")
            return [], {}, None

    def _detect_changes(
        self, json_files: list[Path], existing_corpus: list, existing_manifest: dict, bm25: BM25PlusIndex | None
    ) -> tuple[set, set, dict]:
        """신규/수정/삭제된 source_id를 감지합니다."""
        current_sids = {f.stem for f in json_files}
        current_hashes = {f.stem: _file_hash(f) for f in json_files}
        existing_sids = {
            doc.get(DataFields.METADATA, {}).get(MetadataFields.SOURCE_ID)
            for doc in existing_corpus
            if doc.get(DataFields.METADATA, {}).get(MetadataFields.SOURCE_ID)
        }
        stored_hashes = existing_manifest.get("file_hashes", {})

        if not existing_corpus or bm25 is None:
            modified_sids = current_sids
        else:
            modified_sids = {
                sid for sid, h in current_hashes.items() if h != stored_hashes.get(sid) or sid not in existing_sids
            }
        deleted_sids = existing_sids - current_sids
        return modified_sids, deleted_sids, current_hashes

    def _update_corpus(
        self, existing_corpus: list, modified_sids: set, deleted_sids: set, json_files: list[Path]
    ) -> list:
        """코퍼스를 증분 업데이트합니다."""
        sids_to_remove = modified_sids | deleted_sids
        updated = [
            doc
            for doc in existing_corpus
            if doc.get(DataFields.METADATA, {}).get(MetadataFields.SOURCE_ID) not in sids_to_remove
        ]
        new_raw_data = []
        for f in json_files:
            if f.stem in modified_sids:
                with open(f, encoding="utf-8") as file_obj:
                    try:
                        new_raw_data.append(json.load(file_obj))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 파싱 오류 ({f.name}): {e}")
        updated.extend(self._flatten_data(new_raw_data))
        return updated

    def _build_and_save_index(self, corpus_data: list, current_hashes: dict) -> tuple[list, BM25PlusIndex | None]:
        """토큰화, 인덱스 빌드, 캐시 저장을 수행합니다."""
        tokenized_corpus = [self.tokenizer.tokenize(doc.get(DataFields.CONTENT, "")) for doc in corpus_data]
        valid_indices = [i for i, tokens in enumerate(tokenized_corpus) if tokens]
        if not valid_indices:
            logger.warning("토큰화된 유효 데이터가 없습니다.")
            return [], None

        final_corpus = [corpus_data[i] for i in valid_indices]
        final_tokenized = [tokenized_corpus[i] for i in valid_indices]

        bm25 = BM25PlusIndex()
        bm25.build(final_tokenized)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        bm25.save(self.cache_dir)

        with open(self.cache_dir / _CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(final_corpus, f, ensure_ascii=False, separators=(",", ":"))
        with open(self.cache_dir / _MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump({"docs": len(final_corpus), "file_hashes": current_hashes}, f)

        logger.info(f"통합 인덱스 증분 업데이트 및 저장 완료: {len(final_corpus)} docs")
        return final_corpus, bm25

    def build_or_load(self) -> tuple[list[dict], BM25PlusIndex | None]:
        """
        가공된 데이터를 로드하여 BM25 인덱스를 빌드함.
        신규/수정/삭제된 파일만 부분 감지하여 인덱스를 증분 업데이트(Incremental Update)합니다.
        """
        json_files = self._get_all_json_files()
        if not json_files:
            logger.warning(f"데이터가 없습니다: {self.data_dir} 에 JSON 파일이 없습니다.")
            return [], None

        try:
            existing_corpus, existing_manifest, bm25 = self._load_existing_cache()
            modified_sids, deleted_sids, current_hashes = self._detect_changes(
                json_files, existing_corpus, existing_manifest, bm25
            )

            if not modified_sids and not deleted_sids and existing_corpus and bm25 is not None:
                logger.info(f"통합 인덱스 로드 완료 (Cache - 변경사항 없음): {len(existing_corpus)} docs")
                return existing_corpus, bm25

            logger.info(f"증분 인덱스 업데이트 시작 (수정/추가: {len(modified_sids)}개, 삭제: {len(deleted_sids)}개)")
            corpus_data = self._update_corpus(existing_corpus, modified_sids, deleted_sids, json_files)

            if not corpus_data:
                logger.warning("유효한 텍스트 데이터가 없어 인덱스를 생성할 수 없습니다.")
                return [], None

            return self._build_and_save_index(corpus_data, current_hashes)

        except Exception as e:
            logger.error(f"증분 인덱스 로드 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            return [], None
