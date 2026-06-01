"""리랭커 GPU 장애 → CPU 폴백 → GPU 복구 라이프사이클 실증 스크립트 (Issue #25 / DoD C).

DoD: "리랭커 GPU 오류 모사 시 5분(=RERANKER_GPU_RECOVERY_INTERVAL_SEC=300초) 이내 정상 복구".

실 GPU가 없는 환경에서도 검증 가능하도록, CrossEncoder 모델의 predict 를 mock 하여
CUDA OOM 오류를 1회 주입(fault injection)한다. 그 뒤 서킷브레이커가
  ① GPU 오류 감지 → CPU 폴백 (warning 로그)
  ② recovery_interval 경과 후 다음 호출에서 GPU 복구 시도 (info 로그)
  ③ GPU predict 성공 → 서킷브레이커 해제 (info 로그)
하는 전 과정을 **실제 production 로그 라인**으로 타임스탬프와 함께 캡처한다.

빠른 검증:   python scripts/demo_reranker_gpu_recovery.py --interval 5
실증(5분):   python scripts/demo_reranker_gpu_recovery.py            # 기본 = 운영값 300초
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def _setup_logging() -> Path:
    """src.core.reranker 의 실제 로그를 콘솔 + 파일로 타임스탬프와 함께 캡처."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"reranker_gpu_recovery_{int(time.time())}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    reranker_logger = logging.getLogger("src.core.reranker")
    reranker_logger.setLevel(logging.INFO)
    reranker_logger.addHandler(file_handler)
    reranker_logger.addHandler(console)
    reranker_logger.propagate = False
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="리랭커 GPU 복구 실증")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="GPU 복구 인터벌(초). 미지정 시 운영 설정값(RERANKER_GPU_RECOVERY_INTERVAL_SEC)을 사용.",
    )
    args = parser.parse_args()

    log_path = _setup_logging()

    from langchain_core.documents import Document

    from src.common.config import settings
    from src.core.reranker import CrossEncoderReranker

    interval = args.interval if args.interval is not None else settings.RERANKER_GPU_RECOVERY_INTERVAL_SEC
    settings.RERANKER_GPU_RECOVERY_INTERVAL_SEC = interval

    print("=" * 72)
    print("🧪 리랭커 GPU 장애 → CPU 폴백 → GPU 복구 실증 (Issue #25 / DoD C)")
    print(f"   복구 인터벌: {interval}초  (운영 기본값 = 300초 = 5분)")
    print(f"   로그 파일  : {log_path}")
    print("=" * 72)

    CrossEncoderReranker.reset_instance()
    reranker = CrossEncoderReranker.get_instance(device="cuda")  # GPU 환경 가정
    reranker._original_device = "cuda"

    # CUDA OOM 1회 주입: 1번째 호출 성공(정상 가동) → 2번째 호출 OOM → 이후 성공
    state = {"calls": 0, "fail_on": 2}

    def mock_predict(pairs, batch_size=None):
        state["calls"] += 1
        if state["calls"] == state["fail_on"]:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [0.8 for _ in pairs]

    mock_model = patch("src.core.reranker.CrossEncoder").start()
    mock_model.predict = mock_predict

    docs = [
        Document(page_content="첫 번째 문서", metadata={"chunk_id": "1"}),
        Document(page_content="두 번째 문서", metadata={"chunk_id": "2"}),
    ]

    with patch.object(reranker, "_load_model", return_value=mock_model):
        # --- 단계 1: 정상 GPU 추론 ---
        print("\n[1] 정상 GPU 추론 수행...")
        reranker.rerank(query="질문", documents=docs, top_k=2, threshold=0.0)
        print(f"    device={reranker.device} (정상)")

        # --- 단계 2: GPU OOM 오류 주입 → CPU 폴백 ---
        print("\n[2] GPU OOM 오류 주입 → CPU 폴백 전환...")
        fault_wall = datetime.now()
        reranker.rerank(query="질문", documents=docs, top_k=2, threshold=0.0)
        fault_time = CrossEncoderReranker._gpu_failure_time
        assert reranker.device == "cpu", "CPU 폴백 실패"
        assert fault_time is not None, "장애 시각 미기록"
        print(f"    device={reranker.device} (폴백 완료), 장애 시각 기록됨")

        # --- 단계 3: 인터벌 경과 대기 ---
        # 서킷브레이커는 `경과 >= interval` 조건에서만 복구하므로, 경계를 넘기기 위한 최소 마진만 둔다.
        print(f"\n[3] 복구 인터벌 {interval}초 경과 대기 중...")
        time.sleep(interval + 0.3)

        # --- 단계 4: 다음 호출에서 GPU 복구 ---
        print("\n[4] 인터벌 경과 후 재호출 → GPU 복구 시도...")
        reranker.rerank(query="질문", documents=docs, top_k=2, threshold=0.0)
        recovery_wall = datetime.now()
        assert reranker.device == "cuda", "GPU 복구 실패"
        assert CrossEncoderReranker._gpu_failure_time is None, "서킷브레이커 미해제"
        print(f"    device={reranker.device} (복구 완료)")

    patch.stopall()

    elapsed = (recovery_wall - fault_wall).total_seconds()
    # 서킷브레이커는 설계상 interval 이전엔 복구하지 않는다(조기 복구 = 레이스 위험).
    # 따라서 정상 동작 = "interval 경과 직후 첫 호출에서 복구". 호출 오버헤드 마진 5초 허용.
    fire_margin = elapsed - interval  # 인터벌 경과 후 실제 복구까지의 추가 지연
    healthy = 0 <= fire_margin <= 5
    print("\n" + "=" * 72)
    print("📊 결과")
    print(f"   장애 발생       : {fault_wall.isoformat(timespec='seconds')}")
    print(f"   복구 완료       : {recovery_wall.isoformat(timespec='seconds')}")
    print(f"   복구 인터벌(설정): {interval}초  (= 운영 300초 = 5분)")
    print(f"   장애→복구 소요   : {elapsed:.1f}초")
    print(f"   인터벌 경과 후 복구 발사 지연: {fire_margin:.1f}초")
    verdict = "✅ 정상" if healthy else "❌ 비정상"
    print(f"   판정: {verdict}  (인터벌 경과 직후 첫 호출에서 GPU 복구)")
    print(f"   로그 파일       : {log_path}")
    print("=" * 72)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
