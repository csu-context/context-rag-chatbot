"""
프롬프트 최적화 검증 테스트
"""

from src.core.prompts import RAG_SYSTEM_PROMPT


def test_prompt_structure():
    """프롬프트 구조 검증 - 필수 요소 포함 여부 확인"""
    required_elements = [
        "사내 규정 전문 어시스턴트",  # 페르소나
        "팩트 체크",  # 핵심 목표
        "출처 제시",  # 핵심 목표
        "근거 최우선",  # 답변 원칙
        "<Context>",  # 컨텍스트 태그
    ]

    for elem in required_elements:
        assert elem in RAG_SYSTEM_PROMPT, f"필수 요소 누락: '{elem}'"


def test_chain_structure():
    """체인 구조 및 필수 라이브러리 임포트 검증"""
    from src.core import chains
    from src.core.chains import get_rag_chain

    assert callable(get_rag_chain), "get_rag_chain은 호출 가능한 함수여야 합니다"

    required_imports = [
        "get_rag_chain",
        "ChatPromptTemplate",
        "RunnableLambda",
    ]

    imports = dir(chains)
    for imp in required_imports:
        assert imp in imports, f"필수 객체 임포트 누락: {imp}"
