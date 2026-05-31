"""동의어 과적합 어블레이션 실행 스크립트.

용도: Phase 1 과적합 제거 결정(학칙 매핑 삭제·조대 유지)을 BM25 검색 지표로 정량 검증.
실행: python scripts/ablation_overfit.py   (모델/API/임베딩 불필요)
상세 로직: src/eval/overfit_ablation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.overfit_ablation import main

if __name__ == "__main__":
    main()
