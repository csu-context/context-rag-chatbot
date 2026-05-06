import json
import logging
import pickle

from kiwipiepy import Kiwi
from rank_bm25 import BM25Plus

from src.utils.paths import BM25_CACHE_FILE, PROCESSED_DATA_DIR, SYNONYMS_FILE

logger = logging.getLogger(__name__)


class BM25Manager:
    """
    키워드 기반 검색(BM25)을 관리하는 클래스.
    가공된 JSON 데이터를 로드하여 인덱스를 빌드하고, 형태소 분석 기반의 키워드 검색을 수행함.
    """

    def __init__(self, data_dir=PROCESSED_DATA_DIR):
        """
        BM25 매니저 초기화.

        Args:
            data_dir (Path): 가공된 JSON 파일들이 위치한 디렉토리 경로.
        """
        self.data_dir = data_dir
        self.kiwi = Kiwi()
        self.bm25 = None
        self.corpus_data = []

        # 중앙 관리되는 캐시 파일 경로 사용
        self.cache_path = BM25_CACHE_FILE

        # 동의어 사전 로드
        self.synonyms = self._load_synonyms()

        self.load_index()

    def _load_synonyms(self) -> dict:
        """
        외부 JSON 파일에서 동의어 사전을 로드함.

        Returns:
            dict: {줄임말: 정식명칭} 구조의 동의어 딕셔너리.
        """
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
        """
        입력 텍스트의 줄임말 등을 정식 명칭으로 치환함.

        Args:
            text (str): 치환 전 원본 텍스트.

        Returns:
            str: 동의어가 치환된 텍스트.
        """
        if not self.synonyms:
            return text
        for k, v in self.synonyms.items():
            text = text.replace(k, v)
        return text

    def _tokenizer(self, text: str) -> list[str]:
        """
        한국어 형태소 분석을 통해 의미 있는 토큰(명사, 용언 등)만 추출함.

        Args:
            text (str): 분석할 원본 텍스트.

        Returns:
            list[str]: 정제된 토큰 리스트.
        """
        if not text:
            return []
        text = self._apply_synonyms(text)

        # N: 명사, V: 용언(동사/형용사), S: 외국어/숫자 추출 및 1글자 노이즈 제거
        tokens = [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith(("N", "V", "S")) and len(t.form) > 1]
        return tokens

    def _get_all_json_files(self) -> list:
        """
        데이터 디렉토리 내의 모든 JSON 파일 리스트를 반환함.

        Returns:
            list[Path]: JSON 파일 경로 리스트.
        """
        return list(self.data_dir.glob("*.json"))

    def _should_rebuild_index(self, json_files: list) -> bool:
        """
        캐시 파일의 유효성을 검사하여 인덱스 재빌드 여부를 결정함.

        Args:
            json_files (list): 현재 디렉토리에 존재하는 소스 JSON 파일 리스트.

        Returns:
            bool: 재빌드가 필요하면 True, 아니면 False.
        """
        if not self.cache_path.exists():
            return True

        # 소스 파일 중 하나라도 캐시보다 최신이면 재빌드 필요
        last_mtime = max((f.stat().st_mtime for f in json_files), default=0)
        return last_mtime > self.cache_path.stat().st_mtime

    def load_index(self):
        """
        가공된 데이터를 로드하여 BM25 인덱스를 빌드함.
        캐시가 유효하면 캐시를 로드하고, 그렇지 않으면 신규 빌드함.
        """
        json_files = self._get_all_json_files()

        if not json_files:
            logger.warning(f"데이터가 없습니다: {self.data_dir} 에 JSON 파일이 없습니다.")
            self.bm25 = None
            self.corpus_data = []
            return

        try:
            # 캐시 로드 시도
            if not self._should_rebuild_index(json_files):
                with open(self.cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.bm25 = cached_data["bm25"]
                    self.corpus_data = cached_data["corpus_data"]
                logger.info(f"통합 인덱스 로드 완료 (Cache): {len(self.corpus_data)} docs")
                return

            # 신규 빌드
            logger.info(f"신규 통합 인덱스 빌드 시작 ({len(json_files)} files)")
            self.corpus_data = []
            for json_file in json_files:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.corpus_data.extend(data)
                    else:
                        self.corpus_data.append(data)

            # 모든 문서를 토큰화하여 BM25 인덱스 생성
            tokenized_corpus = [self._tokenizer(doc.get("content", "")) for doc in self.corpus_data]
            self.bm25 = BM25Plus(tokenized_corpus)

            # 빌드된 인덱스 캐싱
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "corpus_data": self.corpus_data}, f)
            logger.info(f"통합 인덱스 빌드 및 저장 완료: {len(self.corpus_data)} docs")

        except Exception as e:
            logger.error(f"인덱스 로드 중 오류 발생: {e}")
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

        # BM25 점수 계산
        scores = self.bm25.get_scores(tokenized_query)
        if not any(scores):
            return []

        # 상위 N개 인덱스 추출
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        top_scores = [scores[i] for i in top_indices]

        # 타 검색 엔진과의 결합을 위한 점수 정규화 (Min-Max Scaling)
        s_max, s_min = max(top_scores), min(top_scores)
        denom = s_max - s_min if s_max != s_min else 1.0
        normalized = [(s - s_min) / denom for s in top_scores]

        results = []
        for rank, (idx, norm_score) in enumerate(zip(top_indices, normalized, strict=False)):
            doc = self.corpus_data[idx]
            if return_scores:
                # 하이브리드 검색을 위한 메타데이터 추가
                results.append({**doc, "_bm25_score": round(norm_score, 4), "_rank": rank + 1})
            else:
                results.append(doc)
        return results


if __name__ == "__main__":
    manager = BM25Manager()
    if manager.bm25:
        logger.info("BM25 통합 검색 엔진이 준비되었습니다.")
    else:
        logger.warning("검색 가능한 데이터가 없습니다.")
