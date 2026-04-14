# src/tests/check_torch.py
import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
print(f"MPS 사용 가능 여부 (macOS): {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("사용 중인 장치: Apple Silicon MPS")
else:
    print("현재 CPU 모드로 동작 중입니다.")
