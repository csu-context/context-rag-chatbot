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

    def test_build_cache_query_empty_history(self, pipeline):
        """대화 이력이 없을 때 캐시 쿼리가 원본 쿼리와 동일하게 빌드되는지 확인"""
        query = "테스트 질문"
        cache_query = pipeline._build_cache_query(query, [])
        assert cache_query == query

    def test_build_cache_query_with_history(self, pipeline):
        """대화 이력이 존재할 때 캐시 쿼리가 히스토리 텍스트와 결합되는지 확인"""
        query = "두 번째 질문"
        history = [
            {"role": "user", "content": "첫 번째 질문"},
            {"role": "assistant", "content": "첫 번째 답변"},
        ]
        cache_query = pipeline._build_cache_query(query, history)
        assert "[History]" in cache_query
        assert "user: 첫 번째 질문" in cache_query
        assert "assistant: 첫 번째 답변" in cache_query
        assert "[Current Query]" in cache_query
        assert "두 번째 질문" in cache_query

    def test_build_cache_query_sliding_window(self, pipeline):
        """대화 이력이 3개 쌍(6개 메시지)을 초과할 때 최근 6개만 결합되는지 검증"""
        query = "새 질문"
        history = [
            {"role": "user", "content": "버려질 질문 1"},
            {"role": "assistant", "content": "버려질 답변 1"},
            {"role": "user", "content": "유지될 질문 2"},
            {"role": "assistant", "content": "유지될 답변 2"},
            {"role": "user", "content": "유지될 질문 3"},
            {"role": "assistant", "content": "유지될 답변 3"},
            {"role": "user", "content": "유지될 질문 4"},
            {"role": "assistant", "content": "유지될 답변 4"},
        ]
        cache_query = pipeline._build_cache_query(query, history)
        assert "버려질 질문 1" not in cache_query
        assert "유지될 질문 2" in cache_query
        assert "유지될 질문 4" in cache_query

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
        """대화 이력이 존재할 때도 캐시를 정상적으로 조회 및 저장하는지 검증"""
        # pipeline 피스처 내부에 모킹된 캐시 가져오기
        mock_cache = pipeline.cache
        mock_cache.get.return_value = None

        history = [{"role": "user", "content": "질문"}]

        # RAG pipeline.stream 실행
        input_data = {"question": "후속 질문", "k": 1, "final_k": 1, "history": history}

        # 모의 함수들 준비
        mock_retriever.search.return_value = []
        mock_reranker.rerank_with_timeout.return_value = MagicMock(documents=[], scores=[])
        mock_llm.stream.return_value = ["답변"]

        # generator 실행완료 처리
        list(pipeline.stream(input_data))

        # 캐시의 get과 add가 호출되었음을 검증
        mock_cache.get.assert_called_once()
        mock_cache.add.assert_called_once()
