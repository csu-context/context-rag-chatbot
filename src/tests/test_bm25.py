import json
import tempfile
import unittest
from pathlib import Path

from src.vector_db.bm25_manager import BM25Manager


class TestBM25Manager(unittest.TestCase):
    def setUp(self):
        """테스트용 임시 디렉토리 및 여러 데이터 파일 세팅"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        # 파일 1: 조선대 관련
        data1 = [
            {
                "content": "조선대학교 휴학 신청 기간은 3월부터입니다.",
                "metadata": {"src_name": "manual1.pdf", "pg_num": 1},
            }
        ]
        with open(self.data_dir / "data1.json", "w", encoding="utf-8") as f:
            json.dump(data1, f)

        # 파일 2: 복학 및 장학금 관련
        data2 = [
            {
                "content": "복학 신청 방법은 홈페이지를 참조하세요.",
                "metadata": {"src_name": "manual2.pdf", "pg_num": 2},
            },
            {
                "content": "성적 장학금 지급 기준 안내입니다.",
                "metadata": {"src_name": "manual2.pdf", "pg_num": 3},
            },
        ]
        with open(self.data_dir / "data2.json", "w", encoding="utf-8") as f:
            json.dump(data2, f)

        self.manager = BM25Manager(data_dir=self.data_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_file_loading(self):
        """여러 JSON 파일이 하나로 통합 로드되는지 확인"""
        self.assertIsNotNone(self.manager.bm25)
        self.assertEqual(len(self.manager.corpus_data), 3)

    def test_search_basic(self):
        results = self.manager.get_top_n("휴학", n=1)
        self.assertEqual(len(results), 1)
        self.assertIn("휴학", results[0]["content"])

    def test_synonyms_from_file(self):
        """외부 synonyms.json 기반 동의어 치환 확인"""
        # synonyms.json에 '조대' -> '조선대학교'가 있다고 가정
        results = self.manager.get_top_n("조대", n=1)
        self.assertEqual(len(results), 1)
        self.assertIn("조선대학교", results[0]["content"])

    def test_no_result(self):
        results = self.manager.get_top_n("전혀없는단어", n=1)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
