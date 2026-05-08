import json

import pytest

from src.vector_db.bm25_manager import BM25Manager


@pytest.fixture
def bm25_manager(tmp_path):
    """테스트용 임시 디렉토리 및 데이터 파일 세팅"""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    # 파일 1: 조선대 관련
    data1 = [
        {
            "content": "조선대학교 휴학 신청 기간은 3월부터입니다.",
            "metadata": {"src_name": "manual1.pdf", "pg_num": 1},
        }
    ]
    with open(data_dir / "data1.json", "w", encoding="utf-8") as f:
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
    with open(data_dir / "data2.json", "w", encoding="utf-8") as f:
        json.dump(data2, f)

    return BM25Manager(data_dir=data_dir)


def test_multi_file_loading(bm25_manager):
    """여러 JSON 파일이 하나로 통합 로드되는지 확인"""
    assert bm25_manager.bm25 is not None
    assert len(bm25_manager.corpus_data) == 3


def test_search_basic(bm25_manager):
    results = bm25_manager.get_top_n("휴학", n=1)
    assert len(results) == 1
    assert "휴학" in results[0]["content"]


def test_synonyms_from_file(bm25_manager):
    """외부 synonyms.json 기반 동의어 치환 확인"""
    results = bm25_manager.get_top_n("조대", n=1)
    assert len(results) == 1
    assert "조선대학교" in results[0]["content"]


def test_no_result(bm25_manager):
    results = bm25_manager.get_top_n("전혀없는단어", n=1)
    assert len(results) == 0


def test_hierarchical_loading(tmp_path):
    """계층형 데이터(children, parent_text)가 평탄화되어 모두 로드되는지 확인"""
    data_dir = tmp_path / "hierarchical_processed"
    data_dir.mkdir()

    # 실제 데이터와 유사한 계층 구조
    data = [
        {
            "parent_id": "p1",
            "parent_text": "보안 규정 서문입니다.",
            "metadata": {"source_id": "doc1", "src_name": "security.pdf"},
            "children": [
                {
                    "chunk_id": "c1",
                    "text": "제1조 목적: 정보 자산 보호.",
                    "metadata": {"parent_id": "p1"},
                }
            ],
        }
    ]

    with open(data_dir / "hierarchical.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    manager = BM25Manager(data_dir=data_dir)
    contents = [doc["content"] for doc in manager.corpus_data]

    assert "제1조 목적: 정보 자산 보호." in contents
    assert "보안 규정 서문입니다." in contents
