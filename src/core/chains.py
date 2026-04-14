import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.prompts import SYSTEM_PROMPT

# 환경 변수 로드
load_dotenv()


def get_rag_chain(retriever):
    # 1. 모델 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=1.0, google_api_key=os.getenv("GOOGLE_API_KEY"), safety_settings=None
    )

    # 2. 컨텍스트 포맷팅 함수
    def format_docs(docs):
        formatted = []
        for doc in docs:
            # 메타데이터에서 소스와 페이지 정보를 안전하게 추출
            source = doc.metadata.get("source", "알 수 없는 파일")
            page = doc.metadata.get("page", "-")
            content = f"내용: {doc.page_content}\n출처: [{source}, p.{page}]"
            formatted.append(content)
        return "\n\n".join(formatted)

    # 3. 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])

    # 4. RAG 체인 구성
    rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm

    return rag_chain
