import json
import pickle
import logging
from rank_bm25 import BM25Plus
from kiwipiepy import Kiwi
from src.utils.paths import PROCESSED_DATA_DIR, SYNONYMS_FILE, BM25_CACHE_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Manager:
    def __init__(self, data_dir=PROCESSED_DATA_DIR):
        self.data_dir = data_dir
        self.kiwi = Kiwi()
        self.bm25 = None
        self.corpus_data = []
        
        # [표준화] 중앙 관리되는 캐시 파일 경로 사용
        self.cache_path = BM25_CACHE_FILE

        # 동의어 사전 로드
        self.synonyms = self._load_synonyms()

        self.load_index()

    def _load_synonyms(self) -> dict:
        """외부 JSON 파일에서 동의어 사전을 로드"""
        if not SYNONYMS_FILE.exists():
            logger.warning(f"동의어 파일을 찾을 수 없습니다: {SYNONYMS_FILE} (빈 사전 사용)")
            return {}
        
        try:
            with open(SYNONYMS_FILE, "r", encoding="utf-8") as f:
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

    def _tokenizer(self, text: str) -> list:
        if not text:
            return []
        text = self._apply_synonyms(text)
        tokens = [
            t.form for t in self.kiwi.tokenize(text)
            if t.tag.startswith(("N", "V", "S")) and len(t.form) > 1
        ]
        return tokens

    def _get_all_json_files(self) -> list:
        """processed 디렉토리 내의 모든 JSON 파일 리스트 반환"""
        return list(self.data_dir.glob("*.json"))

    def _should_rebuild_index(self, json_files: list) -> bool:
        """파일 추가/삭제/수정 여부를 확인하여 재빌드 필요성 판단"""
        if not self.cache_path.exists():
            return True
        last_mtime = max((f.stat().st_mtime for f in json_files), default=0)
        return last_mtime > self.cache_path.stat().st_mtime

    def load_index(self):
        json_files = self._get_all_json_files()

        if not json_files:
            logger.warning(f"데이터가 없습니다: {self.data_dir} 에 JSON 파일이 없습니다.")
            self.bm25 = None
            self.corpus_data = []
            return

        try:
            if not self._should_rebuild_index(json_files):
                with open(self.cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    self.bm25 = cached_data["bm25"]
                    self.corpus_data = cached_data["corpus_data"]
                logger.info(f"통합 인덱스 로드 완료 (Cache): {len(self.corpus_data)} docs")
                return

            logger.info(f"신규 통합 인덱스 빌드 시작 ({len(json_files)} files)")
            self.corpus_data = []
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.corpus_data.extend(data)
                    else:
                        self.corpus_data.append(data)

            tokenized_corpus = [
                self._tokenizer(doc.get("content", ""))
                for doc in self.corpus_data
            ]
            self.bm25 = BM25Plus(tokenized_corpus)

            # [보완] 캐시 디렉토리 자동 생성 후 저장
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "corpus_data": self.corpus_data
                }, f)
            logger.info(f"통합 인덱스 빌드 및 저장 완료: {len(self.corpus_data)} docs")

        except Exception as e:
            logger.error(f"인덱스 로드 중 오류 발생: {e}")
            self.bm25 = None
            self.corpus_data = []

    def get_top_n(self, query: str, n: int = 5, return_scores: bool = False) -> list:
        if not self.bm25 or not self.corpus_data:
            return []
        tokenized_query = self._tokenizer(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        if not any(scores):
            return []
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        top_scores = [scores[i] for i in top_indices]
        s_max, s_min = max(top_scores), min(top_scores)
        denom = s_max - s_min if s_max != s_min else 1.0
        normalized = [(s - s_min) / denom for s in top_scores]
        results = []
        for rank, (idx, norm_score) in enumerate(zip(top_indices, normalized)):
            doc = self.corpus_data[idx]
            if return_scores:
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
