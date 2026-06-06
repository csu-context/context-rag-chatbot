import torch


def get_torch_device() -> str:
    """시스템 하드웨어 상태에 따른 최적의 PyTorch 디바이스 문자열을 반환합니다.
    (NVIDIA/AMD GPU -> 'cuda', Apple Silicon -> 'mps', 일반 환경 -> 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
