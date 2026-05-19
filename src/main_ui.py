import time

import streamlit as st

# --- 1. 페이지 설정 (가장 상단에 위치) ---
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

    # [목표 1] 모델 설정
    st.subheader("모델 설정")
    selected_model = st.selectbox(
        "사용할 LLM 모델 선택",
        ["claude-3-5-sonnet-20240620", "gemma2"],
        index=0
    )

    # [목표 1] 검색 결과 개수(K) 조절 슬라이더
    st.subheader("검색 설정")
    k_value = st.slider(
        "검색할 문서 조각 개수 (K)",
        min_value=1,
        max_value=10,
        value=4,  # 기본값
        step=1
    )

    # [목표 3] 파일 업로더 구현 (전처리 파이프라인 연동용)
    st.subheader("문서 관리")
    uploaded_files = st.file_uploader(
        "매뉴얼 파일 업로드 (PDF, DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"총 {len(uploaded_files)}개의 파일이 선택되었습니다.")
        # 추후 여기에 전처리 파이프라인 연동 로직 추가 예정
        if st.button("문서 DB화 시작"):
            with st.spinner("문서를 분석하고 DB에 저장 중입니다..."):
                time.sleep(2)
                st.success("문서 DB화 완료!")

    st.markdown("---")

    # [목표 1] DB 초기화 버튼 배치
    st.subheader("시스템 관리")
    if st.button("벡터 DB 초기화", help="저장된 모든 문서 데이터를 삭제합니다."):
        st.warning("정말로 초기화하시겠습니까? (로직 미구현)")
        # 추후 여기에 ChromaDB 초기화 로직 추가 예정
        st.session_state.messages = []  # 대화 기록도 초기화

# --- 4. 메인 채팅창 (Main Chat) 구성 ---
st.title("기업 매뉴얼 기반 지능형 챗봇")
st.markdown(f"현재 모델: **{selected_model}** | 검색 개수(K): **{k_value}**")
st.markdown("---")

# [목표 2] st.chat_message를 사용한 대화 히스토리 시각화
# 세션 상태에 저장된 모든 메시지를 순회하며 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. 사용자 입력 처리 및 대화 로직 ---
# 사용자가 채팅 입력창에 입력을 하고 엔터를 치면 실행됨
if prompt := st.chat_input("궁금한 점을 입력해 주세요."):
    # 5-1. 사용자 메시지를 화면에 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5-2. 사용자 메시지를 세션 상태(히스토리)에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5-3. 챗봇의 답변을 생성하는 로직 (추후 백엔드 엔진 연동)
    with st.chat_message("assistant"), st.spinner("생각 중..."):
        time.sleep(1)  # RAG 연동 전 가짜 대기 시간

        # 가로 길이 120자를 넘지 않도록 괄호를 활용해 문자열을 분리해 줬어
        response = (
            f"'{prompt}'에 대한 답변입니다. (추후 RAG 백엔드와 연동되어 "
            "실제 매뉴얼 내용을 기반으로 답변합니다.)\n\n"
            "**인용 출처:** [운영매뉴얼.pdf, 12p]"
        )
        st.markdown(response)

    # 5-4. 챗봇의 답변을 세션 상태(히스토리)에 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
