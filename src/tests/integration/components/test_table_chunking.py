from src.common.constants import MetadataFields
from src.processing.chunking import create_parent_child_chunks


def test_table_chunking():
    """거대한 마크다운 표 분할 및 메타데이터 테스트"""
    dummy_metadata = {
        MetadataFields.SOURCE_ID: "TEST_TABLE_001",
        MetadataFields.SRC_NAME: "test_table_manual.md",
        MetadataFields.DOC_TYPE: "markdown",
        MetadataFields.PG_NUM: 1,
    }

    # 의도적으로 자식 청크 사이즈(400자)를 아득히 초과하는 거대한 표 생성
    massive_table_rows = "\n".join(
        [f"| {i} | 테스트 데이터 {i} | 길이가 꽤 긴 텍스트를 넣어서 용량을 늘립니다. |" for i in range(1, 15)]
    )

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

    # 검증
    assert len(hierarchical_data) > 0

    # 표가 포함된 자식 청크가 있는지 확인
    found_table_chunk = False
    for parent in hierarchical_data:
        for child in parent["children"]:
            if child["metadata"].get(MetadataFields.IS_TABLE):
                found_table_chunk = True
                # 표 헤더가 포함되어 있는지 확인 (복제 로직)
                assert "| ID | 항목명 | 상세 설명 |" in child["text"]
                assert "|---|---|---|" in child["text"]

    assert found_table_chunk, "표가 포함된 청크를 찾을 수 없습니다."
