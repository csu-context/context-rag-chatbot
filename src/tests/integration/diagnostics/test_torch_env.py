import torch


def test_torch_basic():
    """PyTorch 기본 설치 및 동작 확인"""
    assert torch.__version__ is not None

    # 텐서 생성 테스트
    x = torch.ones(5)
    assert x.sum() == 5


def test_device_availability():
    """가용한 컴퓨팅 장치 정보 확인 (CPU는 항상 가용)"""
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    # 장치 정보가 boolean으로 정상 반환되는지 확인
    assert isinstance(is_cuda, bool)
    assert isinstance(is_mps, bool)
