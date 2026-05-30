import json
from unittest.mock import patch

from src.vector_db.bm25_index import BM25PlusIndex
from src.vector_db.bm25_manager import BM25Manager


def test_bm25_index_build_save_load(tmp_path):
    corpus = [["hello", "world"], ["hello", "test", "index"], ["scipy", "matrix"]]
    index = BM25PlusIndex()
    index.build(corpus)

    assert index.corpus_size == 3
    assert index.vocab["hello"] == 0
    assert index.vocab["matrix"] == 5

    # Scores
    scores = index.get_scores(["hello"])
    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[2] == 0

    # Save
    index.save(tmp_path)
    assert (tmp_path / "tf_matrix.npz").exists()
    assert (tmp_path / "arrays.npz").exists()
    assert (tmp_path / "vocab.json").exists()

    # Load
    loaded = BM25PlusIndex.load(tmp_path)
    assert loaded.corpus_size == 3
    assert loaded.vocab["hello"] == 0

    scores_loaded = loaded.get_scores(["hello"])
    assert (scores == scores_loaded).all()


def test_bm25_manager_synonyms_and_tokenizer(tmp_path):
    # Synonyms file
    syn_file = tmp_path / "synonyms.json"
    syn_file.write_text(json.dumps({"AI": "인공지능", "chatbot": "챗봇"}), encoding="utf-8")

    with patch("src.vector_db.bm25_manager.SYNONYMS_FILE", syn_file):
        manager = BM25Manager(data_dir=tmp_path, cache_dir=tmp_path / "cache")
        # _apply_synonyms
        assert manager._apply_synonyms("AI chatbot 개발") == "인공지능 챗봇 개발"
        # Tokenizer
        tokens = manager._tokenizer("AI chatbot 개발에 대하여")
        assert "인공" in tokens
        assert "지능" in tokens


def test_bm25_manager_indexing_and_retrieve(tmp_path):
    # Setup data
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()

    doc1 = {"text": "인공지능 챗봇 기술은 매우 유용합니다.", "metadata": {"source_id": "doc1", "category": "IT"}}
    doc2 = {
        "text": "조선대학교 학칙 제1조 목적 규정을 준수합니다.",
        "metadata": {"source_id": "doc2", "category": "규정"},
    }

    (data_dir / "doc1.json").write_text(json.dumps(doc1), encoding="utf-8")
    (data_dir / "doc2.json").write_text(json.dumps(doc2), encoding="utf-8")

    # Load index (first build)
    manager = BM25Manager(data_dir=data_dir, cache_dir=cache_dir)
    assert manager.bm25 is not None
    assert len(manager.corpus_data) == 2

    # Retrieve
    results = manager.retrieve("챗봇 기술")
    assert len(results) >= 1
    assert "챗봇" in results[0]["content"]

    # Filtered Retrieve
    results_filtered = manager.retrieve("학칙", metadata_filter={"category": "규정"})
    assert len(results_filtered) == 1
    assert "조선대학교" in results_filtered[0]["content"]

    # Incremental update (no changes)
    manager_again = BM25Manager(data_dir=data_dir, cache_dir=cache_dir)
    assert manager_again.bm25 is not None

    # Incremental update (modify one file)
    doc1_mod = {
        "text": "인공지능 챗봇 엔진 및 NLP 기술은 매우 유용합니다.",
        "metadata": {"source_id": "doc1", "category": "IT"},
    }
    (data_dir / "doc1.json").write_text(json.dumps(doc1_mod), encoding="utf-8")

    manager_mod = BM25Manager(data_dir=data_dir, cache_dir=cache_dir)
    assert len(manager_mod.corpus_data) == 2
    assert "NLP" in manager_mod.corpus_data[0]["content"] or "NLP" in manager_mod.corpus_data[1]["content"]
