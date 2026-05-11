import json
import os
import sys
from pathlib import Path

from src.utils.paths import BASE_DIR

# 기본 품질 게이트 임계값 설정
DEFAULT_THRESHOLD_PCT = 5.0  # 정확도 하락 허용치 (%)
DEFAULT_TARGET_LATENCY = 5.0  # 목표 레이턴시 (초)


def compare_metrics(baseline_path: Path, current_path: Path):
    """
    베이스라인과 현재 평가 결과를 비교하여 품질 게이트 통과 여부를 판정합니다.
    """
    threshold_pct = float(os.getenv("GATE_THRESHOLD_PCT", DEFAULT_THRESHOLD_PCT))
    target_latency = float(os.getenv("GATE_TARGET_LATENCY", DEFAULT_TARGET_LATENCY))

    if not baseline_path.exists():
        print(f"❌ 베이스라인 파일을 찾을 수 없습니다: {baseline_path}")
        return False
    if not current_path.exists():
        print(f"❌ 현재 평가 파일을 찾을 수 없습니다: {current_path}")
        return False

    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(current_path, encoding="utf-8") as f:
        current = json.load(f)

    print("\n" + "=" * 50)
    print("      MLOps Quality Gate 검증 리포트")
    print("=" * 50)

    all_passed = True

    # 1. 레이턴시 검증
    curr_latency = current.get("avg_latency_sec", 0.0)
    base_latency = baseline.get("avg_latency_sec", 0.0)

    latency_status = "✅ PASS" if curr_latency <= target_latency else "🚨 FAIL"
    if curr_latency > target_latency:
        all_passed = False

    print("[LATENCY]")
    print(f"   - Baseline: {base_latency:.2f}s")
    print(f"   - Current : {curr_latency:.2f}s")
    print(f"   - Target  : {target_latency:.2f}s")
    print(f"   - Status  : {latency_status}")
    print("-" * 50)

    # 2. 정확도 지표 비교
    # 설정에 따라 비교할 지표를 환경변수에서 가져올 수 있음 (쉼표로 구분)
    metrics_env = os.getenv("GATE_METRICS", "faithfulness,answer_relevancy")
    metrics_to_check = [m.strip().lower() for m in metrics_env.split(",")]

    for metric in metrics_to_check:
        b_val = baseline["scores"].get(metric, 0)
        c_val = current["scores"].get(metric, 0)

        diff = c_val - b_val
        # 하락폭 계산 (음수면 하락)
        drop_pct = (diff / b_val * 100) if b_val > 0 else 0

        status = "✅ PASS" if drop_pct >= -threshold_pct else "❌ FAIL"
        if drop_pct < -threshold_pct:
            all_passed = False

        print(f"[{metric.upper()}]")
        print(f"   - Baseline: {b_val:.4f}")
        print(f"   - Current : {c_val:.4f}")
        print(f"   - Change  : {diff:+.4f} ({drop_pct:+.2f}%)")
        print(f"   - Status  : {status} (Threshold: -{threshold_pct}%)")
        print("-" * 50)

    # 총평
    if all_passed:
        print("\n🎉 품질 게이트 통과! 모든 지표가 허용 범위 내에 있습니다.")
    else:
        print("\n🚨 품질 게이트 실패! 기준 미달 항목이 존재합니다.")
        sys.exit(1)  # CI/CD 파이프라인 중단을 위해 비정상 종료 코드 반환

    print("="*50 + "\n")
    return all_passed



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_eval.py <baseline_json> <current_json>")
        sys.exit(1)

    # BASE_DIR 기준으로 경로 resolve (절대 경로 입력 시에도 대응)
    baseline_file = (BASE_DIR / sys.argv[1]).resolve()
    current_file = (BASE_DIR / sys.argv[2]).resolve()

    success = compare_metrics(baseline_file, current_file)
    sys.exit(0 if success else 1)
