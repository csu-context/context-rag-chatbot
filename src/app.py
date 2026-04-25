import streamlit as st
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from src.vector_db.chroma_manager import ChromaDBManager
from src.core.chains import get_rag_chain
from src.utils.paths import ensure_directories

# --- 1. 페이지 설정 (가장 상단에 위치) ---
st.set_page_config(
    page_title="기업 매뉴얼 챗봇 (1차 통합)",
    layout="wide",
)

# 필수 디렉토리 확인 및 생성
ensure_directories()

# --- 2. RAG 시스템 초기화 (캐싱) ---
@st.cache_resource
def initialize_rag_system():
    """ChromaDB와 RAG 체인을 초기화하고 캐싱합니다."""
    # ChromaDBManager 초기화 (기본 컬렉션 사용)
    db_manager = ChromaDBManager(collection_name="rag_collection")
    # RAG 체인 생성 (vector_db 인스턴스 주입)
    rag_chain = get_rag_chain(db_manager)
    return db_manager, rag_chain

try:
    db_manager, rag_chain = initialize_rag_system()
except Exception as e:
    st.error(f"시스템 초기화 중 오류 발생: {e}")
    st.stop()

# --- 3. 초기 세션 상태 설정 (대화 기록 유지용) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. 사이드바 (Sidebar) 구성 ---
with st.sidebar:
    st.title("🛠️ 설정 및 관리")

    st.subheader("🔍 검색 설정")
    k_value = st.slider(
        "검색할 문서 조각 개수 (K)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    st.divider()
    st.subheader("📊 데이터베이스 상태")
    count = db_manager.get_count()
    st.write(f"현재 저장된 청크 수: **{count}**")
    
    if st.button("🔄 상태 새로고침"):
        st.rerun()

    st.divider()
    st.subheader("📄 문서 관리 (준비 중)")
    uploaded_files = st.file_uploader(
        "새로운 매뉴얼 업로드",
        type=["pdf", "docx", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.info("파일 업로드 기능은 현재 전처리 파이프라인과 통합 중입니다.")

# --- 5. 메인 채팅창 구성 ---
st.title("🤖 지능형 사내 규정 어시스턴트")
st.markdown("사내 매뉴얼 및 규정 문서를 기반으로 답변을 생성합니다. (1차 프로토타입)")
st.markdown("---")

# 대화 히스토리 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. 사용자 입력 및 RAG 실행 ---
if prompt := st.chat_input("규정에 대해 궁금한 점을 물어보세요."):
    # 사용자 메시지 표시 및 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 어시스턴트 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("관련 규정을 분석하여 답변을 생성하고 있습니다..."):
            try:
                # 실제 RAG 체인 호출
                # k_value를 입력 변수로 전달하여 검색 정확도 조절
                response = rag_chain.invoke({"question": prompt, "k": k_value})
                
                # 답변 출력 및 저장
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "죄송합니다. 내부 시스템 오류로 답변을 생성할 수 없습니다."
                })

# --- 7. 푸터 ---
st.markdown("---")
st.caption("© 2026 Context-RAG-Chatbot Team. 모든 답변은 등록된 문서에 근거합니다.")
