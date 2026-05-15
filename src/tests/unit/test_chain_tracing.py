import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.core.chains import get_rag_chain
from src.utils.logger import TracingLogger


@pytest.mark.asyncio
async def test_rerank_rank_change_logging(tmp_path):
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
                    mock_model = MagicMock()
                    mock_model.ainvoke = AsyncMock()
                    mock_model.model_name = "test-model"
                    mock_model.temperature = 0.5  # JSON 직렬화 가능하도록 구체적인 타입(float) 할당

                    mock_response = AIMessage(content="answer")

                    async def mock_astream(*args, **kwargs):
                        yield mock_response

                    mock_model.astream = mock_astream
                    mock_model.ainvoke.return_value = mock_response
                    mock_llm_inst.get_model.return_value = mock_model
                    mock_factory.return_value = mock_llm_inst

                    chain = get_rag_chain(mock_db)

                    # Async generator 소진
                    async for _ in chain.astream({"question": "test", "k": 2}):
                        pass

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
            # llm_params 추출 확인
            assert "llm_params" in gen_step
            assert gen_step["llm_params"].get("model") == "test-model"
            assert gen_step["llm_params"].get("temperature") == 0.5
            assert "latency_ms" in gen_step
