import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.prompts import RAG_SYSTEM_PROMPT
from src.utils.citation import format_citations

# 환경 변수 로드
load_dotenv()


def get_rag_chain(retriever):
    # 1. 모델 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        safety_settings=None,
    )

    # 2. 컨텍스트 포맷팅 함수
    def format_docs(docs):
        formatted = []
        for doc in docs:
            # 표준 메타데이터 규격(src_name, pg_num) 우선 사용
            source = doc.metadata.get("src_name") or doc.metadata.get("source", "알 수 없는 파일")
            page = doc.metadata.get("pg_num") or doc.metadata.get("page", "-")
            content = f"내용: {doc.page_content}\n출처: [{source}, p.{page}]"
            formatted.append(content)
        return "\n\n".join(formatted)

    # 3. 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM_PROMPT), ("human", "{question}")])

    # 4. RAG 체인 구성 (원본 문서를 보존하기 위해 RunnableParallel 사용)
    def combine_answer_and_citations(input_dict):
        answer = input_dict["answer"].content
        citations = format_citations(input_dict["docs"])
        return f"{answer}{citations}"

    rag_chain = (
        RunnablePassthrough.assign(docs=retriever)
        .assign(context=lambda x: format_docs(x["docs"]))
        .assign(answer=prompt | llm)
    ) | combine_answer_and_citations

    return rag_chain
