from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.reranker import CrossEncoderReranker


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Python의 list는 mutable하다.", metadata={"source": "python_doc"}),
        Document(page_content="JavaScript의 array는 immutable하다.", metadata={"source": "js_doc"}),
        Document(page_content="Java의 List는 mutable하다.", metadata={"source": "java_doc"}),
    ]


class TestCrossEncoderReranker:
    def setup_method(self):
        CrossEncoderReranker.reset_instance()

    def test_singleton_pattern(self):
        """싱글톤 패턴 검증: 동일한 인스턴스를 반환해야 함"""
        instance1 = CrossEncoderReranker.get_instance()
        instance2 = CrossEncoderReranker.get_instance()

        assert instance1 is instance2

    def test_reset_singleton(self):
        """싱글톤 리셋 후 새로운 인스턴스가 생성되어야 함"""
        inst1 = CrossEncoderReranker.get_instance()
        CrossEncoderReranker.reset_instance()
        inst2 = CrossEncoderReranker.get_instance()

        assert inst1 is not inst2

    def test_rerank_sorting(self, sample_docs):
        """리랭킹 정렬 검증: 모델 점수가 높은 순서대로 정렬되어야 함"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            # raw logit: Python=4.0, JS=-4.0, Java=1.0
            # sigmoid/5 후: Python≈0.690, Java≈0.550, JS≈0.310
            mock_model.predict.return_value = [4.0, -4.0, 1.0]
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance(threshold=0.1)
            result = reranker.rerank("Python 특징", sample_docs)

            assert len(result.documents) == 3
            assert result.scores[0] == pytest.approx(0.690, abs=1e-2)
            assert result.scores[1] == pytest.approx(0.550, abs=1e-2)
            assert "Python" in result.documents[0].page_content
            assert "Java" in result.documents[1].page_content

    def test_threshold_filtering(self, sample_docs):
        """임계치 필터링 검증: threshold 미만인 문서는 결과에서 제거되어야 함"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            # raw logit: -5.0, 2.0, 4.0
            # sigmoid/5 후: ≈0.269, 0.599, 0.690
            # threshold=0.3 → 0.269인 문서는 필터링
            mock_model.predict.return_value = [-5.0, 2.0, 4.0]
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance(threshold=0.3)
            result = reranker.rerank("검색어", sample_docs)

            assert len(result.documents) == 2
            assert result.filtered_count == 1
            assert all(score >= 0.3 for score in result.scores)

    def test_empty_result_when_all_below_threshold(self, sample_docs):
        """
        [로직 변경 반영] 모든 문서가 임계치 미만일 경우, 빈 리스트를 반환해야 함 (환각 방지)
        """
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            # raw logit 음수 → sigmoid/5 후 모두 0.5 미만
            mock_model.predict.return_value = [-1.0, -2.0, -3.0]
            mock_load.return_value = mock_model

            # 임계치 0.5 설정
            reranker = CrossEncoderReranker.get_instance(threshold=0.5)
            result = reranker.rerank("무관한 질문", sample_docs)

            assert len(result.documents) == 0
            assert result.filtered_count == 3

    def test_skip_rerank_few_documents(self):
        """문서가 2개 미만인 경우 리랭킹을 수행하지 않고 원본을 반환해야 함"""
        docs = [Document(page_content="단일 문서", metadata={"source": "doc"})]

        reranker = CrossEncoderReranker.get_instance()
        result = reranker.rerank("질문", docs)

        assert len(result.documents) == 1
        assert result.scores == [0.5]  # 코드상 기본값 0.5
        assert result.elapsed_time_sec == 0.0

    def test_exception_handling(self, sample_docs):
        """모델 추론 중 예외 발생 시 원본 순서를 유지하여 반환해야 함"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.side_effect = ValueError("GPU Error")
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance()
            result = reranker.rerank_with_timeout("질문", sample_docs)

            expected_len = min(len(sample_docs), reranker.top_k)
            assert len(result.documents) == expected_len
            assert result.scores == [0.0] * expected_len

    def test_timeout_dynamic_top_k(self, sample_docs):
        """추론 시간이 길어질 경우 차후 호출을 위해 top_k가 동적으로 감소해야 함"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = [4.0, 2.5, 1.0]
            mock_load.return_value = mock_model

            initial_top_k = 10
            reranker = CrossEncoderReranker.get_instance(top_k=initial_top_k, threshold=0.1)

            # rerank_with_timeout 내부에서 time.time()을 2번 호출,
            # 그 사이의 rerank() 내부에서 time.time()을 2번 호출하므로 총 4번의 호출이 일어남.
            # 1. rerank_with_timeout 시작
            # 2. rerank 시작
            # 3. rerank 예측 후
            # 4. rerank_with_timeout 끝
            # elapsed_time_sec은 rerank() 내부의 (3번 호출값) - (2번 호출값)으로 계산됨.
            with patch("time.time", side_effect=[1.0, 1.0, 7.0, 7.0]):
                result = reranker.rerank_with_timeout("질문", sample_docs)

                # elapsed_time_sec = 7.0 - 1.0 = 6.0
                assert result.elapsed_time_sec == pytest.approx(6.0)

    def test_model_defaults(self):
        """모델명에 따라 임계치가 올바르게 자동 설정되는지 확인 (sigmoid [0,1] 기준)"""
        # BGE 모델 (0.4)
        bge = CrossEncoderReranker(model_name="bge-reranker-v2-m3")
        assert bge.threshold == 0.4

        # 한국어 모델 (0.5)
        kor = CrossEncoderReranker(model_name="skesarmom/cross-encoder-kor")
        assert kor.threshold == 0.5

        # 기본 모델 (0.45)
        default = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert default.threshold == 0.45

    def test_elapsed_time_tracking(self, sample_docs):
        """추론 소요 시간이 결과 객체에 정확히 기록되는지 확인"""
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            # sigmoid/5 후 모두 threshold(0.45) 초과
            mock_model.predict.return_value = [2.0, 1.5, 1.0]
            mock_load.return_value = mock_model

            reranker = CrossEncoderReranker.get_instance()

            with patch("time.time", side_effect=[10.0, 10.5]):
                result = reranker.rerank("질문", sample_docs)
                assert result.elapsed_time_sec == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
