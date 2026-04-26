import os
import json
import pickle
import logging
from rank_bm25 import BM25Plus
from konlpy.tag import Okt
from src.utils.paths import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "processed", "data.json")


class BM25Manager:
    def __init__(self, data_path=_DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.okt = Okt()
        self.bm25 = None
        self.corpus_data = []

        self.synonyms = {
            "조선대": "조선대학교",
            "조대": "조선대학교",
        }

        self.load_index()

    def _apply_synonyms(self, text: str) -> str:
        for k, v in self.synonyms.items():
            text = text.replace(k, v)
        return text

    def _tokenizer(self, text: str) -> list:
        if not text:
            return []

        text = self._apply_synonyms(text)

        KEEP_POS = {"Noun", "Verb", "Adjective", "Alpha", "Number"}
        tokens = [
            word for word, pos in self.okt.pos(text, stem=False)
            if pos in KEEP_POS and len(word) > 1
        ]
        return tokens

    def load_index(self):
        pickle_path = self.data_path.replace(".json", "_index.pkl")

        if not os.path.exists(self.data_path):
            logger.warning(f"파일을 찾을 수 없습니다: {self.data_path} → 빈 인덱스로 초기화")
            self.bm25 = None
            self.corpus_data = []
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.corpus_data = json.load(f)

            pkl_is_stale = (
                not os.path.exists(pickle_path)
                or os.path.getmtime(self.data_path) > os.path.getmtime(pickle_path)
            )

            if not pkl_is_stale:
                with open(pickle_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                logger.info(f"인덱스 로드 완료 (Pickle 사용): {len(self.corpus_data)} docs")
            else:
                logger.info("신규 인덱스 빌드를 시작합니다.")
                tokenized_corpus = [
                    self._tokenizer(doc.get("content", ""))
                    for doc in self.corpus_data
                ]
                self.bm25 = BM25Plus(tokenized_corpus)

                with open(pickle_path, "wb") as f:
                    pickle.dump(self.bm25, f)
                logger.info("신규 인덱스 빌드 및 Pickle 저장 완료")

        except Exception as e:
            logger.error(f"인덱스 로드 중 오류 발생: {e}")
            self.bm25 = None
            self.corpus_data = []

    def get_top_n(self, query: str, n: int = 5, return_scores: bool = False) -> list:
        if not self.bm25 or not self.corpus_data:
            return []

        tokenized_query = self._tokenizer(query)
        if not tokenized_query:
            logger.info(f"유효 토큰 없음 (조사/빈 쿼리): '{query}'")
            return []

        scores = self.bm25.get_scores(tokenized_query)

        if max(scores) == 0:
            logger.info(f"매칭 결과 없음: '{query}'")
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

    test_queries = ["휴학 신청 기간", "복학 신청 방법", "성적 장학금", "조선대"]

    print("\n" + "=" * 55)
    print("이슈 #34: BM25 검색 엔진 최종 테스트")
    print("=" * 55)

    for q in test_queries:
        results = manager.get_top_n(q, n=1, return_scores=True)
        print(f"\n질의어: '{q}'")
        if results:
            r = results[0]
            print(f"추출 문장: {r['content']}")
            print(f"출처: {r['metadata'].get('src_name')}  p.{r['metadata'].get('pg_num')}")
            print(f"BM25 점수: {r['_bm25_score']}")
        else:
            print("결과: 검색 결과가 없습니다.")
        print("-" * 55)