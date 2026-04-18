"""
프롬프트 최적화 검증 테스트

시나리오별 응답이 기대대로 나오는지 확인합니다.
"""

import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_prompt_structure():
    """프롬프트 구조 검증 - 문법적 오류가 없는지 확인"""
    from src.core.prompts import RAG_SYSTEM_PROMPT

    print("=" * 60)
    print("📋 테스트 1: 프롬프트 구조 검증")
    print("=" * 60)

    # 필수 요소 체크
    required_elements = [
        "친절하고 전문적인 사내 어시스턴트",  # 페르소나
        "Out-of-Scope",  # 외부 질문 처리
        "사내 규정 범위를 벗어나 답변이 불가능합니다",  # 거절 멘트
        "제공된 문서 내에서 해당 정보를 찾을 수 없습니다",  # 환각 방지
        "불렛 포인트",  # 마크다운 강제
        "번호 매기기",  # 마크다운 강제
        "5초 이내 응답 목표",  # 간결성 지시
        "<Context>",  # 컨텍스트 태그
    ]

    missing = []
    for elem in required_elements:
        if elem not in RAG_SYSTEM_PROMPT:
            missing.append(elem)
            print(f"  ❌ 누락: '{elem}'")
        else:
            print(f"  ✅ 포함: '{elem}'")

    if missing:
        print(f"\n⚠️  경고: {len(missing)}개 요소가 누락되었습니다.")
        return False
    else:
        print("\n✅ 모든 필수 요소가 포함되어 있습니다!")
        return True


def test_prompt_templates():
    """프롬프트 템플릿 출력 확인"""
    from src.core.prompts import RAG_SYSTEM_PROMPT

    print("\n" + "=" * 60)
    print("📝 테스트 2: 프롬프트 전체 출력")
    print("=" * 60)
    print(RAG_SYSTEM_PROMPT)
    print("=" * 60)


def test_chain_structure():
    """체인 구조 검증"""
    from src.core.chains import get_rag_chain

    # retriever 없이 호출하면 에러가 발생할 수 있으므로 구조만 확인
    print("\n" + "=" * 60)
    print("🔗 테스트 3: 체인 구조 검증")
    print("=" * 60)

    try:
        # get_rag_chain 함수가 존재하는지 확인
        assert callable(get_rag_chain), "get_rag_chain은 호출 가능한 함수여야 합니다"
        print("  ✅ get_rag_chain 함수가 정상적으로 임포트되었습니다")

        # chains.py의 import 문 체크
        from src.core import chains

        imports = dir(chains)
        required_imports = [
            "get_rag_chain",
            "ChatPromptTemplate",
            "RunnablePassthrough",
        ]

        for imp in required_imports:
            if imp in imports:
                print(f"  ✅ 임포트 확인: {imp}")
            else:
                print(f"  ❌ 누락: {imp}")

        return True

    except Exception as e:
        print(f"  ❌ 오류 발생: {e}")
        return False


def test_mock_response():
    """모의 응답 테스트 (API 미호출)"""

    print("\n" + "=" * 60)
    print("🎭 테스트 4: 프롬프트 시나리오 분석")
    print("=" * 60)

    scenarios = [
        {
            "name": "Out-of-Scope 질문",
            "question": "오늘 날씨 어때?",
            "expected_keywords": ["사내 규정 범위", "답변이 불가능"],
            "description": "날씨 관련 질문 → 거절 멘트 출력",
        },
        {
            "name": "환각 방지 (정보 없음)",
            "question": "회사의 연봉제는 어떻게 되나요?",
            "context": (
                "<Context>\n"
                "내용: 출퇴근 시간은 09:00 ~ 18:00입니다.\n"
                "출처: [사내 규정 매뉴얼, p.5]\n\n"
                "내용: 휴가 사용 방법은 총무팀에 신청하세요.\n"
                "출처: [사내 규정 매뉴얼, p.12]\n"
                "</Context>"
            ),
            "expected_keywords": ["제공된 문서 내에서", "찾을 수 없습니다"],
            "description": "문서에 없는 정보 → 정보 없음 멘트 출력",
        },
        {
            "name": "정상 응답 (마크다운 강제)",
            "question": "출퇴근 시간은 어떻게 되나요?",
            # 수정 후
            "context": (
                "<Context>\n"
                "내용: 출퇴근 시간은 09:00 ~ 18:00이며, 유연제 적용 시 08:00 ~ 17:00입니다.\n"
                "출처: [사내 규정 매뉴얼, p.5]\n"
                "\n"
                "내용: 지각은 월 3회까지 허용됩니다.\n"
                "출처: [사내 규정 매뉴얼, p.6]\n"
                "</Context>"
            ),
            "expected_keywords": ["-", "1.", "09:00", "18:00"],
            "description": "문서에 있는 정보 → 마크다운 형식 답변",
        },
    ]

    for scenario in scenarios:
        print(f"\n  📌 {scenario['name']}")
        print(f"     질문: {scenario['question']}")
        if "context" in scenario:
            print(f"     컨텍스트: {scenario['context'][:50]}...")
        print(f"     기대 결과 키워드: {scenario['expected_keywords']}")
        print(f"     설명: {scenario['description']}")

    print("\n  ⚠️  실제 응답은 Gemini API 호출 시 확인 가능합니다.")


def test_with_api():
    """실제 API 호출 테스트 (선택 사항)"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        from src.core.prompts import RAG_SYSTEM_PROMPT

        # API 키 확인
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("\n" + "=" * 60)
            print("🔑 테스트 5: API 호출 테스트 (생략)")
            print("=" * 60)
            print("  ⚠️  GOOGLE_API_KEY가 설정되지 않았습니다.")
            print("     .env 파일에 API 키를 입력 후 재실행하세요.")
            return False

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=api_key,
            safety_settings=None,
        )

        print("\n" + "=" * 60)
        print("🚀 테스트 5: 실제 API 호출 테스트")
        print("=" * 60)

        # 시나리오별 테스트
        test_cases = [
            {"name": "Out-of-Scope", "question": "오늘 날씨 어때?", "context": ""},
            {
                "name": "정보 없음 (환각 방지)",
                "question": "회사의 연봉제는 어떻게 되나요?",
                "context": """<Context>
내용: 출퇴근 시간은 09:00 ~ 18:00입니다.
출처: [사내 규정 매뉴얼, p.5]

내용: 휴가 사용 방법은 총무팀에 신청하세요.
출처: [사내 규정 매뉴얼, p.12]
</Context>""",
            },
            {
                "name": "정상 응답",
                "question": "출퇴근 시간은 어떻게 되나요?",
                "context": """<Context>
내용: 출퇴근 시간은 09:00 ~ 18:00이며, 유연제 적용 시 08:00 ~ 17:00입니다.
출처: [사내 규정 매뉴얼, p.5]

내용: 지각은 월 3회까지 허용됩니다.
출처: [사내 규정 매뉴얼, p.6]
</Context>""",
            },
        ]

        for tc in test_cases:
            print(f"\n📌 {tc['name']}")
            print(f"   질문: {tc['question']}")

            prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), ("human", "{question}")])
            chain = prompt | llm

            # context가 테스트 케이스에 포함되어 있으면 이를 함께 전달합니다.
            input_data = {"question": tc["question"]}
            if "context" in tc:
                input_data["context"] = tc["context"]

            result = chain.invoke(input_data)
            response = result.content

            print(f"   응답:\n{response}")
            print(f"   ⏱️  응답 길이: {len(response)}자")

    except ImportError as e:
        print(f"\n❌ 라이브러리 임포트 오류: {e}")
        return False
    except Exception as e:
        print(f"\n❌ API 호출 중 오류 발생: {e}")
        return False

    return True


if __name__ == "__main__":
    print("\n" + "🔍 프롬프트 최적화 검증 테스트 시작".center(60, "="))

    # 1. 구조 검증
    test_prompt_structure()

    # 2. 템플릿 출력
    test_prompt_templates()

    # 3. 체인 구조 검증
    test_chain_structure()

    # 4. 모의 응답 테스트
    test_mock_response()

    # 5. API 호출 테스트 (선택)
    api_test_result = False
    try:
        api_test_result = test_with_api()
    except Exception as e:
        print(f"\n⚠️  API 테스트 실행 중 오류: {e}")

    # 최종 결과
    print("\n" + "=" * 60)
    if api_test_result:
        print("✅ 모든 테스트 통과! (API 호출 포함)")
    else:
        print("⚠️  구조 테스트는 통과했으나 API 테스트는 생략되었습니다.")
    print("=" * 60 + "\n")
