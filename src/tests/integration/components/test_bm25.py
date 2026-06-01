import json

import numpy as np
import pytest

from src.vector_db.bm25_index import BM25PlusIndex
from src.vector_db.bm25_manager import BM25Manager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_manager(tmp_path, json_files: dict) -> BM25Manager:
    """tmp_path 안에 데이터 디렉토리와 캐시 디렉토리를 분리하여 BM25Manager를 생성함."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    cache_dir = tmp_path / "bm25_cache"

    for filename, payload in json_files.items():
        with open(data_dir / filename, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return BM25Manager(data_dir=data_dir, cache_dir=cache_dir)


@pytest.fixture
def bm25_manager(tmp_path):
    """기본 검색 기능 테스트용 매니저"""
    return _make_manager(
        tmp_path,
        {
            "data1.json": [
                {
                    "content": "조선대학교 휴학 신청 기간은 3월부터입니다.",
                    "metadata": {"src_name": "manual1.pdf", "pg_num": 1},
                }
            ],
            "data2.json": [
                {
                    "content": "복학 신청 방법은 홈페이지를 참조하세요.",
                    "metadata": {"src_name": "manual2.pdf", "pg_num": 2},
                },
                {
                    "content": "성적 장학금 지급 기준 안내입니다.",
                    "metadata": {"src_name": "manual2.pdf", "pg_num": 3},
                },
            ],
        },
    )


# ---------------------------------------------------------------------------
# 기존 기능 검증
# ---------------------------------------------------------------------------


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
    manager = _make_manager(
        tmp_path,
        {
            "hierarchical.json": [
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
        },
    )

    contents = [doc["content"] for doc in manager.corpus_data]
    assert "제1조 목적: 정보 자산 보호." in contents
    assert "보안 규정 서문입니다." in contents


def test_corpus_fields_slim(tmp_path):
    """corpus_data에 content·metadata만 보존되는지 확인 (불필요 필드 제거 검증)"""
    manager = _make_manager(
        tmp_path,
        {
            "slim.json": [
                {
                    "chunk_id": "c1",
                    "text": "필드 슬림화 테스트 문서입니다.",
                    "parent_text": "삭제되어야 할 필드",
                    "extra_field": "삭제되어야 할 필드",
                    "metadata": {"src_name": "slim.pdf"},
                }
            ]
        },
    )

    assert len(manager.corpus_data) == 1
    doc = manager.corpus_data[0]

    # 보존되어야 할 필드
    assert "content" in doc
    assert "metadata" in doc
    assert doc["chunk_id"] == "c1"  # RRF 중복 제거용 최상위 chunk_id

    # 제거되어야 할 불필요 필드
    assert "text" not in doc
    assert "parent_text" not in doc
    assert "extra_field" not in doc


# ---------------------------------------------------------------------------
# 직렬화 라운드트립 검증
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path):
    """인덱스를 저장한 뒤 재로드해도 검색 결과가 동일한지 확인"""
    manager = _make_manager(
        tmp_path,
        {
            "rt.json": [
                {"content": "캐시 저장 및 로드 테스트입니다.", "metadata": {}},
                {"content": "전혀 다른 주제의 문서입니다.", "metadata": {}},
            ]
        },
    )

    query = "캐시"
    results_before = manager.get_top_n(query, n=1, return_scores=True)
    assert len(results_before) == 1

    # 캐시에서 재로드
    manager2 = BM25Manager(data_dir=tmp_path / "processed", cache_dir=tmp_path / "bm25_cache")
    results_after = manager2.get_top_n(query, n=1, return_scores=True)

    assert len(results_after) == 1
    assert results_before[0]["content"] == results_after[0]["content"]
    assert abs(results_before[0]["_bm25_score"] - results_after[0]["_bm25_score"]) < 1e-4


def test_bm25_index_serialization(tmp_path):
    """BM25PlusIndex.save / load 라운드트립: 점수 배열이 동일한지 확인"""
    corpus = [["휴학", "신청", "기간"], ["복학", "신청", "홈페이지"], ["장학금", "지급", "기준"]]
    index = BM25PlusIndex()
    index.build(corpus)

    cache_dir = tmp_path / "idx_cache"
    index.save(cache_dir)

    loaded = BM25PlusIndex.load(cache_dir)

    query_tokens = ["신청"]
    scores_orig = index.get_scores(query_tokens)
    scores_loaded = loaded.get_scores(query_tokens)

    np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-5)


def test_return_scores_flag(bm25_manager):
    """return_scores=True 시 _bm25_score 및 _rank 필드가 포함되는지 확인"""
    results = bm25_manager.get_top_n("휴학", n=2, return_scores=True)
    assert all("_bm25_score" in r for r in results)
    assert all("_rank" in r for r in results)
    assert results[0]["_rank"] == 1
