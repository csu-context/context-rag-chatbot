import json
import pickle
import logging
from rank_bm25 import BM25Plus
from kiwipiepy import Kiwi
from src.utils.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [표준화] utils.paths에서 정의된 경로를 기반으로 설정
_DEFAULT_DATA_PATH = PROCESSED_DATA_DIR / "data.json"


class BM25Manager:
    def __init__(self, data_path=_DEFAULT_DATA_PATH):
        # 입력받은 data_path가 문자열일 경우 Path 객체로 변환하여 통일
        self.data_path = PROCESSED_DATA_DIR / data_path if isinstance(data_path, str) and not data_path.startswith("/") else data_path
        self.kiwi = Kiwi()
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

        # Kiwi 형태소 분석 (명사, 용언, 외국어, 숫자 추출)
        tokens = [
            t.form for t in self.kiwi.tokenize(text)
            if t.tag.startswith(("N", "V", "S")) and len(t.form) > 1
        ]
        return tokens

    def load_index(self):
        # [표준화] pathlib를 활용한 파일 경로 조작
        pickle_path = self.data_path.with_name(self.data_path.stem + "_index.pkl")

        if not self.data_path.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {self.data_path} → 빈 인덱스로 초기화")
            self.bm25 = None
            self.corpus_data = []
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.corpus_data = json.load(f)

            # [표준화] pathlib.stat()을 활용한 mtime 비교
            pkl_is_stale = (
                not pickle_path.exists()
                or self.data_path.stat().st_mtime > pickle_path.stat().st_mtime
            )

            if not pkl_is_stale:
                with open(pickle_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                logger.info(f"인덱스 로드 완료 (Pickle 사용): {len(self.corpus_data)} docs")
            else:
                logger.info("신규 인덱스 빌드를 시작합니다. (Kiwi 분석기 사용)")
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

        if not any(scores):
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
    # 임시 테스트용 데이터 생성
    import tempfile
    from pathlib import Path

    test_data = [
        {"content": "조선대학교 휴학 신청 기간은 3월부터입니다.", "metadata": {"src_name": "test.pdf", "pg_num": 1}},
        {"content": "복학 신청 방법은 홈페이지를 참조하세요.", "metadata": {"src_name": "test.pdf", "pg_num": 2}},
        {"content": "성적 장학금 지급 기준 안내입니다.", "metadata": {"src_name": "test.pdf", "pg_num": 3}},
    ]

    # 임시 파일 경로를 Path 객체로 관리
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(test_data, tf)
        temp_path = Path(tf.name)

    try:
        print("\n" + "=" * 55)
        print("BM25 엔진 최종 테스트 (Pathlib 표준화 버전)")
        print("=" * 55)

        manager = BM25Manager(data_path=temp_path)
        test_queries = ["휴학 신청 기간", "복학", "장학금", "조대"]

        for q in test_queries:
            results = manager.get_top_n(q, n=1, return_scores=True)
            print(f"\n질의어: '{q}'")
            if results:
                r = results[0]
                print(f"추출 문장: {r['content']}")
                print(f"BM25 점수: {r['_bm25_score']}")
            else:
                print("결과: 검색 결과가 없습니다.")
            print("-" * 55)
    finally:
        if temp_path.exists():
            temp_path.unlink()
        pkl_path = temp_path.with_name(temp_path.stem + "_index.pkl")
        if pkl_path.exists():
            pkl_path.unlink()
