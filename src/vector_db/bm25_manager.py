import hashlib
import json
import logging
import traceback
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from kiwipiepy import Kiwi

from src.common.constants import DataFields, MetadataFields
from src.core.base_retriever import BaseRetriever
from src.utils.paths import BM25_CACHE_DIR, PROCESSED_DATA_DIR, SYNONYMS_FILE
from src.vector_db.bm25_index import BM25PlusIndex

logger = logging.getLogger(__name__)

_CORPUS_FILE = "corpus.json"
_MANIFEST_FILE = "manifest.json"


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


class BM25Manager(BaseRetriever):
    def __init__(self, data_dir=PROCESSED_DATA_DIR, cache_dir=BM25_CACHE_DIR):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.kiwi = Kiwi()
        self.bm25: BM25PlusIndex | None = None
        self.corpus_data: list[dict] = []
        self.synonyms = self._load_synonyms()
        self.load_index()

    def _load_synonyms(self) -> dict:
        if not SYNONYMS_FILE.exists():
            logger.warning(f"동의어 파일을 찾을 수 없습니다: {SYNONYMS_FILE} (빈 사전 사용)")
            return {}
        try:
            with open(SYNONYMS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"동의어 로드 중 오류 발생: {e}")
            return {}

    def _apply_synonyms(self, text: str) -> str:
        if not self.synonyms:
            return text
        for k, v in self.synonyms.items():
            text = text.replace(k, v)
        return text

    STOPWORDS: ClassVar[set[str]] = {
        "대한",
        "대해",
        "위해",
        "통해",
        "경우",
        "또한",
        "모든",
        "의한",
        "따라",
        "기타",
        "사항",
        "있거나",
        "있으며",
        "의하여",
        "관하여",
        "다만",
    }

    def _tokenizer(self, text: str) -> list[str]:
        """한국어 형태소 분석을 통해 의미 있는 토큰(명사, 용언 등)만 추출하고 불용어를 필터링함."""
        if not text:
            return []
        text = self._apply_synonyms(text)
        # N: 명사, V: 용언(동사/형용사), S: 외국어/숫자 추출 및 1글자 노이즈 제거
        tokens = [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith(("N", "V", "S")) and len(t.form) > 1]
        return [tok for tok in tokens if tok not in self.STOPWORDS]

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

    def _get_all_json_files(self) -> list:
        return [f for f in self.data_dir.glob("*.json") if f.name != "manifest.json"]

    def _load_existing_cache(self) -> tuple[list, dict]:
        """기존 캐시 코퍼스와 매니페스트를 로드합니다. 실패 시 빈 값 반환."""
        corpus_path = self.cache_dir / _CORPUS_FILE
        manifest_path = self.cache_dir / _MANIFEST_FILE
        if not (corpus_path.exists() and manifest_path.exists()):
            return [], {}
        try:
            with open(corpus_path, encoding="utf-8") as f:
                corpus = json.load(f)
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            self.bm25 = BM25PlusIndex.load(self.cache_dir)
            return corpus, manifest
        except Exception as cache_err:
            logger.warning(f"기존 캐시 로드 실패 (전체 재구성): {cache_err}")
            return [], {}

    def _detect_changes(
        self, json_files: list, existing_corpus: list, existing_manifest: dict
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

        if not existing_corpus or self.bm25 is None:
            modified_sids = current_sids
        else:
            modified_sids = {
                sid for sid, h in current_hashes.items() if h != stored_hashes.get(sid) or sid not in existing_sids
            }
        deleted_sids = existing_sids - current_sids
        return modified_sids, deleted_sids, current_hashes

    def _update_corpus(self, existing_corpus: list, modified_sids: set, deleted_sids: set, json_files: list) -> list:
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

    def _build_and_save_index(self, current_hashes: dict) -> None:
        """토큰화, 인덱스 빌드, 캐시 저장을 수행합니다."""
        tokenized_corpus = [self._tokenizer(doc.get(DataFields.CONTENT, "")) for doc in self.corpus_data]
        valid_indices = [i for i, tokens in enumerate(tokenized_corpus) if tokens]
        if not valid_indices:
            logger.warning("토큰화된 유효 데이터가 없습니다.")
            self.bm25 = None
            return
        self.corpus_data = [self.corpus_data[i] for i in valid_indices]
        tokenized_corpus = [tokenized_corpus[i] for i in valid_indices]
        self.bm25 = BM25PlusIndex()
        self.bm25.build(tokenized_corpus)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bm25.save(self.cache_dir)
        with open(self.cache_dir / _CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.corpus_data, f, ensure_ascii=False, separators=(",", ":"))
        with open(self.cache_dir / _MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump({"docs": len(self.corpus_data), "file_hashes": current_hashes}, f)
        logger.info(f"통합 인덱스 증분 업데이트 및 저장 완료: {len(self.corpus_data)} docs")

    def load_index(self) -> None:
        """
        가공된 데이터를 로드하여 BM25 인덱스를 빌드함.
        [DataOps] 신규/수정/삭제된 파일만 부분 감지하여 인덱스를 증분 업데이트(Incremental Update)합니다.
        """
        json_files = self._get_all_json_files()
        if not json_files:
            logger.warning(f"데이터가 없습니다: {self.data_dir} 에 JSON 파일이 없습니다.")
            self.bm25 = None
            self.corpus_data = []
            return

        try:
            existing_corpus, existing_manifest = self._load_existing_cache()
            modified_sids, deleted_sids, current_hashes = self._detect_changes(
                json_files, existing_corpus, existing_manifest
            )

            if not modified_sids and not deleted_sids and existing_corpus and self.bm25 is not None:
                self.corpus_data = existing_corpus
                logger.info(f"통합 인덱스 로드 완료 (Cache - 변경사항 없음): {len(self.corpus_data)} docs")
                return

            logger.info(f"증분 인덱스 업데이트 시작 (수정/추가: {len(modified_sids)}개, 삭제: {len(deleted_sids)}개)")
            self.corpus_data = self._update_corpus(existing_corpus, modified_sids, deleted_sids, json_files)

            if not self.corpus_data:
                logger.warning("유효한 텍스트 데이터가 없어 인덱스를 생성할 수 없습니다.")
                self.bm25 = None
                return

            self._build_and_save_index(current_hashes)

        except Exception as e:
            logger.error(f"증분 인덱스 로드 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            self.bm25 = None
            self.corpus_data = []

    def _apply_metadata_filter(self, metadata_filter: dict) -> list[int]:
        """주어진 메타데이터 필터에 부합하는 문서의 인덱스 목록을 반환합니다."""
        return [
            idx
            for idx, doc in enumerate(self.corpus_data)
            if all(doc.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())
        ]

    def get_top_n(
        self,
        query: str,
        n: int = 5,
        return_scores: bool = False,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if not self.bm25 or not self.corpus_data:
            return []

        tokenized_query = self._tokenizer(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # 메타데이터 필터링 적용 및 매칭되는 문서 인덱스 분류
        if metadata_filter:
            matching_indices = self._apply_metadata_filter(metadata_filter)
        else:
            matching_indices = list(range(len(self.corpus_data)))

        if not matching_indices:
            return []

        s_max = float(np.max(scores[matching_indices]))
        denom = s_max if s_max > 0.0 else 1.0

        # 매칭된 인덱스들에 대해서만 스코어 기반 정렬 수행
        matching_scores = scores[matching_indices]
        if not matching_scores.any():
            return []

        n_matching = len(matching_indices)
        if n_matching <= n:
            sorted_sub_indices = np.argsort(matching_scores)[::-1].tolist()
        else:
            top_k_sub = np.argpartition(matching_scores, -n)[-n:]
            sorted_sub_indices = top_k_sub[np.argsort(matching_scores[top_k_sub])[::-1]].tolist()

        top_indices = [matching_indices[i] for i in sorted_sub_indices]
        normalized = [float(scores[i]) / denom for i in top_indices]

        results = []
        for rank, (idx, norm_score) in enumerate(zip(top_indices, normalized, strict=False)):
            doc = self.corpus_data[idx]
            if return_scores:
                results.append({**doc, "_bm25_score": round(norm_score, 4), "_rank": rank + 1})
            else:
                results.append(doc)
        return results

    def retrieve(self, query: str, n: int = 5, metadata_filter: dict | None = None) -> list[dict[str, Any]]:
        """BaseRetriever 인터페이스 구현. BM25 키워드 검색을 실행합니다."""
        return self.get_top_n(query=query, n=n, return_scores=True, metadata_filter=metadata_filter)


if __name__ == "__main__":
    manager = BM25Manager()
    if manager.bm25:
        logger.info("BM25 통합 검색 엔진이 준비되었습니다.")
    else:
        logger.warning("검색 가능한 데이터가 없습니다.")
