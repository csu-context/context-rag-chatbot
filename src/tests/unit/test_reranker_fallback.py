import math
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.reranker import CrossEncoderReranker


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / 5.0))


def test_reranker_cpu_fallback_on_gpu_error():
    # 싱글톤 인스턴스 리셋
    CrossEncoderReranker.reset_instance()

    # 디바이스를 cuda로 강제 설정하여 CUDA 에러 상황 모방
    reranker = CrossEncoderReranker.get_instance(device="cuda")
    assert reranker.device == "cuda"

    # Mock model
    mock_model = MagicMock()

    # 첫 번째 predict 호출 시 CUDA RuntimeError 발생, 두 번째 predict는 정상 처리
    call_count = 0

    def mock_predict(pairs, batch_size=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("CUDA out of memory error in pytorch")
        return [0.8, 0.5]

    mock_model.predict = mock_predict

    # _load_model이 mock_model을 리턴하도록 패치
    with patch.object(reranker, "_load_model", return_value=mock_model):
        docs = [
            Document(page_content="첫 번째 문서", metadata={"chunk_id": "1"}),
            Document(page_content="두 번째 문서", metadata={"chunk_id": "2"}),
        ]

        # rerank 실행
        result = reranker.rerank(query="질문", documents=docs, top_k=2, threshold=0.1)

        # 1. 디바이스가 cpu로 바뀌었는지 검증
        assert reranker.device == "cpu"
        # 2. predict가 총 2번 호출되었는지 검증 (1차 실패, 2차 CPU 폴백 성공)
        assert call_count == 2
        # 3. 결과 문서가 올바르게 재정렬되었는지 검증 (sigmoid 정규화 적용 후 값)
        assert len(result.documents) == 2
        assert result.scores[0] == pytest.approx(_sigmoid(0.8))
