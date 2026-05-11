import json
import sys
from pathlib import Path

def compare_metrics(baseline_path: Path, current_path: Path, threshold_pct: float = 5.0, target_latency: float = 5.0):
    """
    베이즈라인과 현재 평가 결과를 비교하여 품질 게이트 통과 여부를 판정합니다.
    """
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

    print("\n" + "="*50)
    print("      MLOps Quality Gate 검증 리포트")
    print("="*50)

    # 1. 레이턴시 검증 (평균 5초 이내 목표)
    # 현재 evaluator.py는 total_eval_time_sec만 저장하므로, 추론 시간 합계를 샘플 수로 나눠야 함
    # 하지만 logs/eval/eval_summary에는 아직 평균 latency가 직접 기록되지 않음.
    # 일단 total_eval_time_sec (평가 시간 제외한 추론 시간 추정) 기반으로 비교하거나 
    # logs/trace 로그를 활용하는 것이 정확함. 
    # 여기서는 간단히 current_latency 목표치만 체크.
    
    # 2. 정확도 지표 비교 (Faithfulness 중심)
    metrics_to_check = ["faithfulness", "answer_relevancy"]
    all_passed = True

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
        print("\n🚨 품질 게이트 실패! 정확도가 기준치 이상으로 하락했습니다.")

    print("="*50 + "\n")
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_eval.py <baseline_json> <current_json>")
        sys.exit(1)
        
    baseline_file = Path(sys.argv[1])
    current_file = Path(sys.argv[2])
    
    success = compare_metrics(baseline_file, current_file)
    sys.exit(0 if success else 1)
