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

    STOPWORDS = {
        "대한", "대해", "위해", "통해", "경우", "또한", "모든", "의한", "따라",
        "기타", "사항", "있거나", "있으며", "의하여", "관하여", "다만"
    }

    def _tokenizer(self, text: str) -> list[str]:
        """한국어 형태소 분석을 통해 의미 있는 토큰(명사, 용언 등)만 추출하고 불용어를 필터링함."""
        if not text:
            return []
        text = self._apply_synonyms(text)
        # N: 명사, V: 용언(동사/형용사), S: 외국어/숫자 추출 및 1글자 노이즈 제거
        tokens = [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith(("N", "V", "S")) and len(t.form) > 1]
        return [tok for tok in tokens if tok not in self.STOPWORDS]

    def _get_all_json_files(self) -> list:
        return list(self.data_dir.glob("*.json"))

    def _should_rebuild_index(self, json_files: list) -> bool:
        """manifest.json 타임스탬프를 기준으로 재빌드 여부를 결정함."""
        manifest_path = self.cache_dir / _MANIFEST_FILE
        if not manifest_path.exists():
            return True
        last_mtime = max((f.stat().st_mtime for f in json_files), default=0)
        return last_mtime > manifest_path.stat().st_mtime

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

    def load_index(self):
        """
        가공된 데이터를 로드하여 BM25 인덱스를 빌드함.
        캐시가 유효하면 numpy/scipy 포맷으로 캐시를 로드하고, 그렇지 않으면 신규 빌드함.
        """
        json_files = self._get_all_json_files()

        if not json_files:
            logger.warning(f"데이터가 없습니다: {self.data_dir} 에 JSON 파일이 없습니다.")
            self.bm25 = None
            self.corpus_data = []
            return

        try:
            if not self._should_rebuild_index(json_files):
                self.bm25 = BM25PlusIndex.load(self.cache_dir)
                with open(self.cache_dir / _CORPUS_FILE, encoding="utf-8") as f:
                    self.corpus_data = json.load(f)
                logger.info(f"통합 인덱스 로드 완료 (Cache): {len(self.corpus_data)} docs")
                return

            # 신규 빌드
            logger.info(f"신규 통합 인덱스 빌드 시작 ({len(json_files)} files)")
            all_raw_data = []
            for json_file in json_files:
                with open(json_file, encoding="utf-8") as f:
                    try:
                        all_raw_data.append(json.load(f))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 파싱 오류 ({json_file.name}): {e}")

            self.corpus_data = self._flatten_data(all_raw_data)

            if not self.corpus_data:
                logger.warning("유효한 텍스트 데이터가 없어 인덱스를 생성할 수 없습니다.")
                return

            tokenized_corpus = [self._tokenizer(doc.get(DataFields.CONTENT, "")) for doc in self.corpus_data]

            valid_indices = [i for i, tokens in enumerate(tokenized_corpus) if tokens]
            if not valid_indices:
                logger.warning("토큰화된 유효 데이터가 없습니다.")
                return

            self.corpus_data = [self.corpus_data[i] for i in valid_indices]
            tokenized_corpus = [tokenized_corpus[i] for i in valid_indices]

            self.bm25 = BM25PlusIndex()
            self.bm25.build(tokenized_corpus)

            # 직렬화 저장 (numpy/scipy 네이티브 포맷)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.bm25.save(self.cache_dir)
            with open(self.cache_dir / _CORPUS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.corpus_data, f, ensure_ascii=False, separators=(",", ":"))
            with open(self.cache_dir / _MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump({"docs": len(self.corpus_data)}, f)

            logger.info(f"통합 인덱스 빌드 및 저장 완료: {len(self.corpus_data)} docs")

        except Exception as e:
            logger.error(f"인덱스 로드 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            self.bm25 = None
            self.corpus_data = []

    def get_top_n(
        self,
        query: str,
        n: int = 5,
        return_scores: bool = False,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        질의어와 가장 유사한 상위 N개의 문서 조각을 반환함.

        Args:
            query (str): 검색할 사용자 질의어.
            n (int): 반환할 결과 개수.
            return_scores (bool): 점수(정규화됨)를 포함하여 반환할지 여부.
            metadata_filter (dict): 선택적인 메타데이터 필터링 조건.

        Returns:
            list[dict]: 검색된 문서 조각 및 메타데이터 리스트.
        """
        if not self.bm25 or not self.corpus_data:
            return []

        tokenized_query = self._tokenizer(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # 메타데이터 필터링 적용 및 매칭되는 문서 인덱스 분류
        matching_indices = []
        for idx, doc in enumerate(self.corpus_data):
            if metadata_filter:
                doc_meta = doc.get("metadata", {})
                match = True
                for k, v in metadata_filter.items():
                    if doc_meta.get(k) != v:
                        match = False
                        break
                if not match:
                    scores[idx] = 0.0
                    continue
            matching_indices.append(idx)

        if not matching_indices:
            return []

        # 매칭되는 문서들의 점수 중 최댓값을 구함 (전역 정규화 스케일러용)
        s_max = float(np.max(scores))
        s_min = 0.0
        denom = s_max - s_min if s_max > 0.0 else 1.0

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
        normalized = [(float(scores[i]) - s_min) / denom for i in top_indices]

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
