import json
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.core.chains import get_rag_chain
from src.utils.logger import TracingLogger


def test_rerank_rank_change_logging(tmp_path):
    """리랭킹 전후 순위 변화가 로그에 정확히 기록되는지 검증"""
    TracingLogger.reset_instance()
    with patch("src.utils.logger.LOGS_DIR", tmp_path):
        # Mock DB
        mock_db = MagicMock()

        # _perform_retrieval 로직 시뮬레이션을 위해 Document 객체로 변환되어 반환되도록 패치
        with patch("src.core.chains._perform_retrieval") as mock_retrieval:
            mock_retrieval.return_value = [
                Document(page_content="doc1", metadata={"chunk_id": "id1"}),
                Document(page_content="doc2", metadata={"chunk_id": "id2"}),
            ]

            # Mock Reranker to swap order
            with patch("src.core.reranker.CrossEncoderReranker.get_instance") as mock_get_reranker:
                mock_reranker = MagicMock()
                mock_reranker.rerank_with_timeout.return_value = MagicMock(
                    documents=[
                        Document(page_content="doc2", metadata={"chunk_id": "id2"}),
                        Document(page_content="doc1", metadata={"chunk_id": "id1"}),
                    ],
                    scores=[0.95, 0.85],
                )
                mock_get_reranker.return_value = mock_reranker

                # Mock LLM
                with patch("src.models.factory.LLMFactory.create_llm") as mock_factory:
                    mock_llm_inst = MagicMock()
                    # chains.py는 이제 llm_instance.invoke()를 직접 호출함
                    mock_response = MagicMock()
                    mock_response.content = "answer"
                    mock_response.model_name = "test-model"
                    mock_response.usage = {"total_tokens": 100}
                    mock_response.latency = 0.5

                    mock_llm_inst.invoke.return_value = mock_response
                    mock_factory.return_value = mock_llm_inst

                    chain = get_rag_chain(mock_db)
                    chain.invoke({"question": "test", "k": 2})
        # 로그 확인
        log_files = list((tmp_path / "trace").glob("*.jsonl"))
        with open(log_files[0], encoding="utf-8") as f:
            log_data = json.loads(f.readline())

            # Reranking step 검증
            rerank_step = next(s for s in log_data["steps"] if s["step"] == "reranking")
            assert rerank_step["rank_change"]["before"] == ["id1", "id2"]
            assert rerank_step["rank_change"]["after"] == ["id2", "id1"]

            # Generation step 검증
            gen_step = next(s for s in log_data["steps"] if s["step"] == "generation")
            assert gen_step["model_name"] == "test-model"
            assert "usage" in gen_step
            assert "latency_ms" in gen_step
