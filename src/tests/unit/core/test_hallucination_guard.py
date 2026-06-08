from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.chains import RAGPipeline


@pytest.fixture
def mock_retriever():
    return MagicMock()


@pytest.fixture
def mock_reranker():
    return MagicMock()


class TestRAGHallucinationGuard:
    def test_reranker_threshold_filtering(self, mock_retriever, mock_reranker):
        """점수가 RERANKER_SIMILARITY_THRESHOLD 미만인 문서들이 올바르게 걸러지는지 검증합니다."""
        # 1. RAG 파이프라인 생성 (임계값 0.4 설정)
        with patch("src.core.chains.settings") as mock_settings:
            mock_settings.RERANKER_SIMILARITY_THRESHOLD = 0.4
            pipeline = RAGPipeline(mock_retriever, llm=MagicMock(), reranker=mock_reranker)

            # 모킹 데이터 준비
            doc_high = Document(page_content="신뢰도 높은 고유 문서", metadata={"chunk_id": "doc_1"})
            doc_low = Document(page_content="신뢰도 매우 낮은 무관 문서", metadata={"chunk_id": "doc_2"})

            # 리랭커 결과 모킹
            mock_result = MagicMock()
            mock_result.documents = [doc_high, doc_low]
            mock_result.scores = [0.85, 0.15]  # 하나는 통과, 하나는 미달
            mock_result.reranker_skipped = False
            mock_reranker.rerank_with_timeout.return_value = mock_result

            # 2. 리랭커 동작 수행
            session = MagicMock()
            final_docs, scores, reranker_skipped = pipeline._do_reranking(
                query="테스트 질문", docs=[doc_high, doc_low], final_k=2, session=session
            )

            # 3. 검증 (점수 0.4 미만인 doc_low는 결과에서 제외되어야 함)
            assert len(final_docs) == 1
            assert final_docs[0].metadata["chunk_id"] == "doc_1"
            assert scores == [0.85]
            assert reranker_skipped is False

    def test_all_filtered_empty_context(self, mock_retriever, mock_reranker):
        """모든 문서 점수가 임계값 미만이라 필터링되면, 빈 컨텍스트 리스트를 안전하게 리턴하는지 검증합니다."""
        with patch("src.core.chains.settings") as mock_settings:
            mock_settings.RERANKER_SIMILARITY_THRESHOLD = 0.4
            pipeline = RAGPipeline(mock_retriever, llm=MagicMock(), reranker=mock_reranker)

            doc_low_1 = Document(page_content="무관 문서 1", metadata={"chunk_id": "doc_1"})
            doc_low_2 = Document(page_content="무관 문서 2", metadata={"chunk_id": "doc_2"})

            mock_result = MagicMock()
            mock_result.documents = [doc_low_1, doc_low_2]
            mock_result.scores = [0.22, 0.35]  # 둘 다 미달
            mock_result.reranker_skipped = False
            mock_reranker.rerank_with_timeout.return_value = mock_result

            session = MagicMock()
            final_docs, scores, reranker_skipped = pipeline._do_reranking(
                query="테스트 질문", docs=[doc_low_1, doc_low_2], final_k=2, session=session
            )

            # 검증 (모두 제외)
            assert len(final_docs) == 0
            assert len(scores) == 0
            assert reranker_skipped is False

    def test_reranker_skipped_fallback_filtering(self, mock_retriever, mock_reranker):
        """리랭커가 스킵되었을 때 초벌 검색 점수 기준(RETRIEVER_FALLBACK_THRESHOLD)으로 비상 필터링되는지 검증합니다."""
        with patch("src.core.chains.settings") as mock_settings:
            mock_settings.RETRIEVER_FALLBACK_THRESHOLD = 0.1
            pipeline = RAGPipeline(mock_retriever, llm=MagicMock(), reranker=mock_reranker)

            doc_high = Document(page_content="초벌 검색 점수 높은 문서", metadata={"chunk_id": "doc_1", "score": 0.25})
            doc_low = Document(page_content="초벌 검색 점수 아주 낮은 무관 문서", metadata={"chunk_id": "doc_2", "score": 0.05})

            mock_result = MagicMock()
            mock_result.documents = [doc_high, doc_low]
            mock_result.scores = [0.25, 0.05]
            mock_result.reranker_skipped = True
            mock_reranker.rerank_with_timeout.return_value = mock_result

            session = MagicMock()
            final_docs, scores, reranker_skipped = pipeline._do_reranking(
                query="테스트 질문", docs=[doc_high, doc_low], final_k=2, session=session
            )

            # 초벌 임계값 0.1 미만인 doc_low는 결과에서 제외되어야 함
            assert len(final_docs) == 1
            assert final_docs[0].metadata["chunk_id"] == "doc_1"
            assert scores == [0.25]
            assert reranker_skipped is True

    def test_reranker_skipped_fallback_filtering_with_rrf(self, mock_retriever, mock_reranker):
        """RRF 점수가 존재하더라도 보존된 vector_score를 바탕으로 올바르게 비상 필터링을 수행하는지 검증합니다."""
        with patch("src.core.chains.settings") as mock_settings:
            mock_settings.RETRIEVER_FALLBACK_THRESHOLD = 0.4
            pipeline = RAGPipeline(mock_retriever, llm=MagicMock(), reranker=mock_reranker)

            # RRF 점수(0.01)는 임계값(0.4) 미만이지만, vector_score(0.65)는 임계값 이상인 문서
            doc_high = Document(
                page_content="RRF는 낮으나 벡터 점수는 높은 문서",
                metadata={"chunk_id": "doc_1", "score": 0.016, "vector_score": 0.65, "rrf_score": 0.016}
            )
            # RRF 점수(0.01)도 낮고 vector_score(0.25)도 임계값 미만인 무관 문서
            doc_low = Document(
                page_content="RRF도 낮고 벡터 점수도 낮은 문서",
                metadata={"chunk_id": "doc_2", "score": 0.012, "vector_score": 0.25, "rrf_score": 0.012}
            )

            mock_result = MagicMock()
            mock_result.documents = [doc_high, doc_low]
            mock_result.scores = [0.016, 0.012]
            mock_result.reranker_skipped = True
            mock_reranker.rerank_with_timeout.return_value = mock_result

            session = MagicMock()
            final_docs, scores, reranker_skipped = pipeline._do_reranking(
                query="테스트 질문", docs=[doc_high, doc_low], final_k=2, session=session
            )

            # vector_score가 0.4 이상인 doc_high만 통과해야 함 (RRF 점수 0.01 기준 필터링 오작동 방지)
            assert len(final_docs) == 1
            assert final_docs[0].metadata["chunk_id"] == "doc_1"
            assert scores == [0.65]
            assert reranker_skipped is True
