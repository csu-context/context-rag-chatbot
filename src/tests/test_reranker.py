from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """CrossEncoderReranker 테스트 스위트"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """각 테스트 전 싱글톤 리셋"""
        CrossEncoderReranker.reset_instance()

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(page_content="Python의 list는 mutable하다.", metadata={"source": "python_doc"}),
            Document(page_content="JavaScript의 array는 immutable하다.", metadata={"source": "js_doc"}),
            Document(page_content="Java의 List는 mutable하다.", metadata={"source": "java_doc"}),
        ]

    def test_singleton_pattern(self):
        """싱글톤 패턴 검증: 동일 인스턴스 반환"""
        instance1 = CrossEncoderReranker.get_instance()
        instance2 = CrossEncoderReranker.get_instance()

        assert instance1 is instance2

    def test_reset_singleton(self):
        """싱글톤 리셋 후 새 인스턴스 생성"""
        inst1 = CrossEncoderReranker.get_instance()
        CrossEncoderReranker.reset_instance()
        inst2 = CrossEncoderReranker.get_instance()

        assert inst1 is not inst2

    def test_rerank_sorting(self, sample_docs):
        """리랭킹 정렬 검증: 높은 점수 문서가 상위"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            # Mock 모델 설정 - .tolist() 메서드 필요
            mock_model = MagicMock()
            mock_scores = MagicMock()
            mock_scores.tolist.return_value = [0.8, 0.2, 0.5]
            mock_model.predict.return_value = mock_scores
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance(threshold=0.1)
            result = reranker.rerank("Python 특징", sample_docs)

            # 정렬 확인 (0.8 > 0.5 > 0.2)
            assert result.scores[0] == pytest.approx(0.8)
            assert result.scores[1] == pytest.approx(0.5)

    def test_threshold_filtering(self, sample_docs):
        """임계치 필터링 검증: threshold 미만 문서 제거"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            # 첫 번째 문서는 임계치(0.3) 미만, 나머지는 초과
            mock_scores = MagicMock()
            mock_scores.tolist.return_value = [0.2, 0.5, 0.6]
            mock_model.predict.return_value = mock_scores
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance(threshold=0.3)
            result = reranker.rerank("Python 특징", sample_docs)

            assert len(result.documents) == 2  # 필터링 후 2개만 남음
            assert result.filtered_count == 1

    def test_min_one_document_guarantee(self, sample_docs):
        """최소 1개 보장: 모든 문서가 threshold 미만일 경우 최고 점수 유지"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            # 모든 문서가 임계치(0.9) 미만
            mock_scores = MagicMock()
            mock_scores.tolist.return_value = [0.1, 0.2, 0.3]
            mock_model.predict.return_value = mock_scores
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance(threshold=0.9)
            result = reranker.rerank("Python 특징", sample_docs)

            assert len(result.documents) == 1  # 최소 1개 보장

    def test_skip_rerank_few_documents(self, sample_docs):
        """검색 결과 부족 시 리랭킹 생략"""
        docs = [sample_docs[0]]  # 1개만

        reranker = CrossEncoderReranker.get_instance()
        result = reranker.rerank("Python 특징", docs)

        assert len(result.documents) == 1
        assert result.filtered_count == 0

    def test_exception_handling(self, sample_docs):
        """예외 발생 시 원본 문서 반환"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("Model Error")
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance()
            result = reranker.rerank("Python 특징", sample_docs)

            assert len(result.documents) == 3  # 원본 유지

    def test_timeout_dynamic_top_k(self, sample_docs):
        """시간 초과 시 동적 top_k 조절"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_scores = MagicMock()
            mock_scores.tolist.return_value = [0.8, 0.5, 0.2]
            mock_model.predict.return_value = mock_scores
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance(top_k=10, threshold=0.1)

            # time.time를 모킹하여 5초 이상 소요되도록 설정 (시작: 1.0, 종료: 6.5)
            with patch("time.time", side_effect=[1.0, 6.5]):
                result = reranker.rerank_with_timeout("Python 특징", sample_docs)

                # result를 검증하여 변수 경고 해결
                assert len(result.documents) == 3

                # top_k가 절반으로 줄어드는지 확인 (10 -> 5)
                assert reranker.top_k == 5

    def test_model_defaults(self):
        """모델별 기본 임계치 설정 검증"""
        # BGE 모델
        bge_reranker = CrossEncoderReranker(model_name="bge-reranker-base")
        assert bge_reranker.threshold == 0.2

        # 한국어 모델
        kor_reranker = CrossEncoderReranker(model_name="skesarmom/cross-encoder-kor")
        assert kor_reranker.threshold == 0.4

        # 기본 모델
        default_reranker = CrossEncoderReranker()
        assert default_reranker.threshold == 0.3

    def test_elapsed_time_tracking(self, sample_docs):
        """추론 시간 측정 검증"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_scores = MagicMock()
            mock_scores.tolist.return_value = [0.8, 0.5]
            mock_model.predict.return_value = mock_scores
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance()

            # time.time를 모킹하여 0.5초 소요되도록 설정 (시작: 1.0, 종료: 1.5)
            with patch("time.time", side_effect=[1.0, 1.5]):
                result = reranker.rerank("Python 특징", sample_docs)

                assert result.elapsed_time_sec == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
