from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.chains import RAGPipeline


class TestRAGPipelineMemory:
    @pytest.fixture
    def mock_retriever(self):
        return MagicMock()

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def mock_reranker(self):
        return MagicMock()

    @pytest.fixture
    def pipeline(self, mock_retriever, mock_llm, mock_reranker):
        """테스트 중 실제 ChromaDB 및 TracingLogger 연결을 차단하고 RAGPipeline 인스턴스를 제공하는 피스처"""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache_class,
            patch("src.core.chains.TracingLogger") as mock_logger_class,
        ):
            mock_cache_class.return_value = MagicMock()
            mock_logger_class.return_value = MagicMock()
            pipeline = RAGPipeline(mock_retriever, mock_llm, mock_reranker)
            yield pipeline

    @patch("src.core.chains.ChatPromptTemplate")
    def test_stream_generation_prompt_binding(self, mock_chat_prompt_template, pipeline):
        """프롬프트 템플릿에 이전 대화 이력과 컨텍스트가 정상 바인딩되는지 검증"""
        mock_session = MagicMock()
        mock_prompt_val = MagicMock()
        mock_chat_prompt_template.from_messages.return_value = mock_prompt_val

        # to_messages() 에 대응하는 모의 리스트
        mock_msg = MagicMock()
        mock_msg.content = "프롬프트 텍스트 예시"
        mock_prompt_val.to_messages.return_value = [mock_msg]

        sample_docs = [
            Document(
                page_content="가장 중요한 사내 규정 내용",
                metadata={"src_name": "rule.pdf", "page": 1},
            )
        ]
        history = [
            {"role": "user", "content": "안녕하세요"},
            {"role": "assistant", "content": "반갑습니다"},
        ]

        # Generator를 소모시켜 연산 실행
        list(pipeline._stream_generation("질문", sample_docs, history, mock_session))

        # from_messages가 호출되었는지 검증
        mock_chat_prompt_template.from_messages.assert_called_once()
        called_args = mock_chat_prompt_template.from_messages.call_args[0][0]

        # system, human(대화이력), ai(대화이력), human(현재질문) 순서 검증
        assert len(called_args) == 4
        assert called_args[0][0] == "system"
        assert called_args[1] == ("human", "안녕하세요")
        assert called_args[2] == ("ai", "반갑습니다")
        assert called_args[3] == ("human", "{question}")

    def test_semantic_cache_with_history(self, pipeline, mock_retriever, mock_llm, mock_reranker):
        """대화 이력이 존재할 때 캐시 조회 및 저장 모두 수행하지 않는지 검증"""
        mock_cache = pipeline.cache

        history = [{"role": "user", "content": "질문"}]
        input_data = {"question": "후속 질문", "k": 1, "final_k": 1, "history": history}

        mock_retriever.search.return_value = []
        mock_reranker.rerank_with_timeout.return_value = MagicMock(documents=[], scores=[])
        mock_llm.stream.return_value = ["답변"]

        list(pipeline.stream(input_data))

        # 이력 있으면 캐시 조회 안 함 (컨텍스트 오염 방지)
        mock_cache.get.assert_not_called()
        # 이력 있으면 캐시 저장 안 함
        mock_cache.add.assert_not_called()
