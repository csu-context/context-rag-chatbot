from langchain_core.prompts import PromptTemplate

# ── [파서 벤치마크 평가 로직 보존] ──────────────────────────────────────────

PARSER_BENCHMARK_PROMPT = PromptTemplate(
    input_variables=["ground_truth", "parsed_result"],
    template="""당신은 문서 파싱 파이프라인의 성능을 평가하는 전문가(LLM-as-a-Judge)입니다.
원본(Ground Truth) 데이터와 파서가 추출한 결과(Parsed Result)를 비교하여,
특히 '표(Table) 구조의 보존 여부'와 '정보의 누락'에 초점을 맞추어 평가해주세요.

[원본 구조/텍스트]
{ground_truth}

[파싱된 결과]
{parsed_result}

다음 항목에 대해 1~5점으로 점수를 매기고, 짧은 이유를 설명해주세요:
1. 표 구조 보존 (Markdown 형태 등으로 행/열이 논리적으로 분리되어 있는가?)
2. 데이터 누락 (원문에 있는 수치나 핵심 텍스트가 누락되지 않았는가?)
3. 가독성 및 노이즈 (불필요한 줄바꿈이나 깨진 문자가 없는가?)
""",
)
