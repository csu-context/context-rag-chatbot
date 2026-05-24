import time
import streamlit as st
from pathlib import Path

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="기업 매뉴얼 챗봇",
    page_icon="🤖",
    layout="wide"  # 넓은 화면 레이아웃 사용
)

# --- 2. 초기 세션 상태 설정 (대화 기록 유지용) ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 히스토리를 저장할 리스트

# --- 3. 사이드바 (Sidebar) 구성 ---
with st.sidebar:
    st.title("설정 및 관리")

    # [모델 설정] 유라가 쓰는 모델들로 구성했어
    st.subheader("모델 설정")
    selected_model = st.selectbox(
        "사용할 LLM 모델 선택",
        ["claude-3-5-sonnet-20240620", "gemma2", "phi3"],
        index=0
    )

    # [검색 설정]
    st.subheader("검색 설정")
    k_value = st.slider(
        "검색할 문서 조각 개수 (K)",
        min_value=1,
        max_value=10,
        value=4,
        step=1
    )

    # [문서 관리]
    st.subheader("문서 관리")
    uploaded_files = st.file_uploader(
        "매뉴얼 파일 업로드 (PDF, DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"총 {len(uploaded_files)}개의 파일이 선택되었습니다.")
        if st.button("문서 DB화 시작"):
            with st.spinner("문서를 분석하고 DB에 저장 중입니다..."):
                time.sleep(2)
                st.success("문서 DB화 완료!")

    st.markdown("---")

    # [시스템 관리]
    st.subheader("시스템 관리")
    if st.button("벡터 DB 초기화", help="저장된 모든 문서 데이터를 삭제합니다."):
        st.warning("정말로 초기화하시겠습니까?")
        st.session_state.messages = []  # 대화 기록 초기화

# --- 4. 메인 채팅창 (Main Chat) 구성 ---
st.title("기업 매뉴얼 기반 지능형 챗봇")
st.markdown(f"현재 모델: **{selected_model}** | 검색 개수(K): **{k_value}**")
st.markdown("---")

# 대화 히스토리 시각화
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. 사용자 입력 처리 및 대화 로직 ---
if prompt := st.chat_input("궁금한 점을 입력해 주세요."):
    # 5-1. 사용자 메시지 표시 및 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5-2. 챗봇 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            time.sleep(1)  # 가짜 대기 시간

            # RAG 연동 전 기본 응답
            response = (
                f"'{prompt}'에 대한 답변입니다. (추후 RAG 백엔드와 연동되어 "
                "실제 매뉴얼 내용을 기반으로 답변합니다.)\n\n"
                "**인용 출처:** [운영매뉴얼.pdf, 12p]"
            )
            st.markdown(response)

    # 5-3. 챗봇 답변 저장
    st.session_state.messages.append({"role": "assistant", "content": response})