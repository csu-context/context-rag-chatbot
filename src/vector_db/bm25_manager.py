import json
import logging
import traceback
from typing import Any

import numpy as np
from kiwipiepy import Kiwi

from src.common.constants import DataFields, MetadataFields
from src.core.base_retriever import BaseRetriever
from src.utils.paths import BM25_CACHE_DIR, PROCESSED_DATA_DIR, SYNONYMS_FILE
from src.vector_db.bm25_index import BM25PlusIndex

logger = logging.getLogger(__name__)

_CORPUS_FILE = "corpus.json"
_MANIFEST_FILE = "manifest.json"


class BM25Manager(BaseRetriever):
    """
    키워드 기반 검색(BM25)을 관리하는 클래스.
    가공된 JSON 데이터를 로드하여 인덱스를 빌드하고, 형태소 분석 기반의 키워드 검색을 수행함.

    인덱스 캐시 구조 (.cache/bm25_v2/):
      tf_matrix.npz  — scipy sparse CSC TF 행렬
      arrays.npz     — numpy: idf 배열, doc_len 배열, 스칼라 파라미터
      vocab.json     — 단어→열 인덱스 매핑
      corpus.json    — 슬림 코퍼스 (content + metadata + chunk_id만 보존)
      manifest.json  — 캐시 유효성 타임스탬프
    """

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

    def _tokenizer(self, text: str) -> list[str]:
        """한국어 형태소 분석을 통해 의미 있는 토큰(명사, 용언 등)만 추출함."""
        if not text:
            return []
        text = self._apply_synonyms(text)
        # N: 명사, V: 용언(동사/형용사), S: 외국어/숫자 추출 및 1글자 노이즈 제거
        return [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith(("N", "V", "S")) and len(t.form) > 1]

    def _get_all_json_files(self) -> list:
        return list(self.data_dir.glob("*.json"))

    def _flatten_data(self, data: Any) -> list[dict]:
        """
        계층형 구조를 평탄화하며, 검색에 필요한 필드만 보존함.
          - content  : 검색 본문 (text / content / parent_text 우선순위)
          - metadata : 출처 메타데이터
          - chunk_id : RRF 중복 제거용 (최상위 필드에 존재할 경우만)
        """
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

    def load_index(self):  # noqa: C901
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
            # 1. 기존 캐시 및 코퍼스 로드 시도
            corpus_path = self.cache_dir / _CORPUS_FILE
            manifest_path = self.cache_dir / _MANIFEST_FILE

            existing_corpus = []
            if corpus_path.exists() and manifest_path.exists():
                try:
                    with open(corpus_path, encoding="utf-8") as f:
                        existing_corpus = json.load(f)
                    self.bm25 = BM25PlusIndex.load(self.cache_dir)
                except Exception as cache_err:
                    logger.warning(f"기존 캐시 로드 실패 (전체 재구성): {cache_err}")
                    existing_corpus = []

            # 현재 폴더에 있는 source_id 목록 추출 (파일명이 source_id임)
            current_sids = {f.stem for f in json_files}

            # 기존 코퍼스의 source_id 목록 추출
            existing_sids = set()
            for doc in existing_corpus:
                sid = doc.get(DataFields.METADATA, {}).get(MetadataFields.SOURCE_ID)
                if sid:
                    existing_sids.add(sid)

            # 2. 변경된 파일 감지 (신규 추가, 수정됨, 삭제됨)
            modified_sids = set()

            # 캐시가 완전히 깨졌거나 로드 실패 시 전체 재구축
            if not existing_corpus or self.bm25 is None:
                modified_sids = current_sids
            else:
                cache_time = manifest_path.stat().st_mtime
                for f in json_files:
                    sid = f.stem
                    # 파일 수정 시각이 캐시 기록 시각보다 최근이거나, 기존 코퍼스에 없는 경우
                    if f.stat().st_mtime > cache_time or sid not in existing_sids:
                        modified_sids.add(sid)

            deleted_sids = existing_sids - current_sids

            # 변경 사항이 전혀 없는 경우
            if not modified_sids and not deleted_sids and existing_corpus and self.bm25 is not None:
                self.corpus_data = existing_corpus
                logger.info(f"통합 인덱스 로드 완료 (Cache - 변경사항 없음): {len(self.corpus_data)} docs")
                return

            logger.info(f"증분 인덱스 업데이트 시작 (수정/추가: {len(modified_sids)}개, 삭제: {len(deleted_sids)}개)")

            # 3. 코퍼스 데이터 증분 업데이트
            # 삭제 및 수정된 기존 데이터 필터링 제거
            sids_to_remove = modified_sids | deleted_sids
            updated_corpus = [
                doc
                for doc in existing_corpus
                if doc.get(DataFields.METADATA, {}).get(MetadataFields.SOURCE_ID) not in sids_to_remove
            ]

            # 신규 및 수정된 데이터 로드 및 추가
            new_raw_data = []
            for f in json_files:
                sid = f.stem
                if sid in modified_sids:
                    with open(f, encoding="utf-8") as file_obj:
                        try:
                            new_raw_data.append(json.load(file_obj))
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON 파싱 오류 ({f.name}): {e}")

            new_flattened = self._flatten_data(new_raw_data)
            updated_corpus.extend(new_flattened)

            self.corpus_data = updated_corpus

            if not self.corpus_data:
                logger.warning("유효한 텍스트 데이터가 없어 인덱스를 생성할 수 없습니다.")
                self.bm25 = None
                return

            # 4. 토큰화 및 인덱스 재생성
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

            # 5. 인덱스 및 코퍼스 캐시 영속화
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.bm25.save(self.cache_dir)

            with open(self.cache_dir / _CORPUS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.corpus_data, f, ensure_ascii=False, separators=(",", ":"))

            with open(self.cache_dir / _MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump({"docs": len(self.corpus_data)}, f)

            logger.info(f"통합 인덱스 증분 업데이트 및 저장 완료: {len(self.corpus_data)} docs")

        except Exception as e:
            logger.error(f"증분 인덱스 로드 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            self.bm25 = None
            self.corpus_data = []

    def get_top_n(self, query: str, n: int = 5, return_scores: bool = False) -> list[dict]:
        """
        질의어와 가장 유사한 상위 N개의 문서 조각을 반환함.

        Args:
            query (str): 검색할 사용자 질의어.
            n (int): 반환할 결과 개수.
            return_scores (bool): 점수(정규화됨)를 포함하여 반환할지 여부.

        Returns:
            list[dict]: 검색된 문서 조각 및 메타데이터 리스트.
        """
        if not self.bm25 or not self.corpus_data:
            return []

        tokenized_query = self._tokenizer(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        if not scores.any():
            return []

        n_docs = len(scores)
        if n_docs <= n:
            top_indices = np.argsort(scores)[::-1].tolist()
        else:
            top_k = np.argpartition(scores, -n)[-n:]
            top_indices = top_k[np.argsort(scores[top_k])[::-1]].tolist()
        top_scores = [float(scores[i]) for i in top_indices]

        # 타 검색 엔진과의 결합을 위한 점수 정규화 (Min-Max Scaling)
        s_max, s_min = max(top_scores), min(top_scores)
        denom = s_max - s_min if s_max != s_min else 1.0
        normalized = [(s - s_min) / denom for s in top_scores]

        results = []
        for rank, (idx, norm_score) in enumerate(zip(top_indices, normalized, strict=False)):
            doc = self.corpus_data[idx]
            if return_scores:
                results.append({**doc, "_bm25_score": round(norm_score, 4), "_rank": rank + 1})
            else:
                results.append(doc)
        return results

    def retrieve(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        """BaseRetriever 인터페이스 구현. BM25 키워드 검색을 실행합니다."""
        return self.get_top_n(query=query, n=n, return_scores=True)


if __name__ == "__main__":
    manager = BM25Manager()
    if manager.bm25:
        logger.info("BM25 통합 검색 엔진이 준비되었습니다.")
    else:
        logger.warning("검색 가능한 데이터가 없습니다.")
