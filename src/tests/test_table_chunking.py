import json
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.common.constants import MetadataFields  # noqa: E402
from src.processing.chunking import create_parent_child_chunks  # noqa: E402


def run_table_chunking_test():
    print("🚀 [테스트] 거대한 마크다운 표 분할 및 메타데이터 테스트 시작\n")

    dummy_metadata = {
        MetadataFields.SOURCE_ID: "TEST_TABLE_001",
        MetadataFields.SRC_NAME: "test_table_manual.md",
        MetadataFields.DOC_TYPE: "markdown",
        MetadataFields.PG_NUM: 1,
    }

    # 의도적으로 자식 청크 사이즈(400자)를 아득히 초과하는 거대한 표 생성
    massive_table_rows = "\n".join([f"| {i} | 테스트 데이터 {i} | 길이가 꽤 긴 텍스트를 넣어서 용량을 늘립니다. |" for i in range(1, 15)])

    sample_text = f"""# 제1장 총칙
## 제3조 (데이터베이스 구조)
아래 표는 시스템의 핵심 데이터베이스 구조를 설명합니다.

| ID | 항목명 | 상세 설명 |
|---|---|---|
{massive_table_rows}

표에 대한 설명이 끝났습니다.
"""

    # 청킹 실행
    hierarchical_data = create_parent_child_chunks(sample_text, dummy_metadata)

    # 결과 출력
    for parent_idx, parent in enumerate(hierarchical_data):
        print(f"📂 Parent Chunk {parent_idx + 1}")
        for child_idx, child in enumerate(parent["children"]):
            is_table = child["metadata"].get(MetadataFields.IS_TABLE, False)
            table_badge = "📊 [표 포함]" if is_table else "📝 [일반 텍스트]"

            print(f"  └─ Child Chunk {child_idx + 1} {table_badge} (길이: {len(child['text'])}자)")
            print(f"     내용 미리보기:\n{child['text']}")
            print("-" * 50)


if __name__ == "__main__":
    run_table_chunking_test()