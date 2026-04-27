import json
import unittest
import tempfile
from pathlib import Path
from src.vector_db.bm25_manager import BM25Manager

class TestBM25Manager(unittest.TestCase):
    def setUp(self):
        """테스트용 임시 데이터 세팅"""
        self.test_data = [
            {"content": "조선대학교 휴학 신청 기간은 3월부터입니다.", "metadata": {"src_name": "test.pdf", "pg_num": 1}},
            {"content": "복학 신청 방법은 홈페이지를 참조하세요.", "metadata": {"src_name": "test.pdf", "pg_num": 2}},
            {"content": "성적 장학금 지급 기준 안내입니다.", "metadata": {"src_name": "test.pdf", "pg_num": 3}},
        ]
        
        # 임시 JSON 파일 생성
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temp_dir.name) / "test_data.json"
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.test_data, f)
            
        self.manager = BM25Manager(data_path=self.data_path)

    def tearDown(self):
        """테스트 종료 후 정리"""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """인덱스 로드 및 코퍼스 데이터 확인"""
        self.assertIsNotNone(self.manager.bm25)
        self.assertEqual(len(self.manager.corpus_data), 3)

    def test_search_basic(self):
        """기본 키워드 검색 확인"""
        results = self.manager.get_top_n("휴학", n=1)
        self.assertEqual(len(results), 1)
        self.assertIn("휴학", results[0]["content"])

    def test_synonyms(self):
        """동의어(조대 -> 조선대학교) 치환 검색 확인"""
        results = self.manager.get_top_n("조대", n=1)
        self.assertEqual(len(results), 1)
        self.assertIn("조선대학교", results[0]["content"])

    def test_no_result(self):
        """검색 결과가 없는 경우 확인"""
        results = self.manager.get_top_n("전혀없는단어", n=1)
        self.assertEqual(len(results), 0)

    def test_empty_query(self):
        """빈 쿼리 또는 조사만 있는 경우 확인"""
        results = self.manager.get_top_n("은 는 이 가", n=1)
        self.assertEqual(len(results), 0)

    def test_score_normalization(self):
        """점수 정규화(0~1 사이) 확인"""
        results = self.manager.get_top_n("장학금", n=1, return_scores=True)
        if results:
            score = results[0]["_bm25_score"]
            self.assertTrue(0 <= score <= 1)

if __name__ == "__main__":
    unittest.main()
