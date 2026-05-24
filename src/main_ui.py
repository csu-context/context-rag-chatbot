import time

import streamlit as st

# ==============================================================================
# 1. 페이지 및 레이아웃 설정
# ==============================================================================
st.set_page_config(
    page_title="기업 매뉴얼 챗봇",
    page_icon="🤖",
    layout="wide"
)

# ==============================================================================
# 2. 전역 세션 상태(Session State) 초기화
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 히스토리 영속성을 위한 리스트 초기화

# ==============================================================================
# 3. 사이드바(Sidebar) UI 컴포넌트 구성
# ==============================================================================
with st.sidebar:
    st.title("설정 및 관리")

    # 3-1. 대화형 인터페이스에 사용될 LLM(Large Language Model) 선택
    st.subheader("모델 설정")
    selected_model = st.selectbox(
        "사용할 LLM 모델 선택",
        ["claude-3-5-sonnet-20240620", "gemma2", "phi3"],
        index=0
    )

    # 3-2. RAG 검색기(Retriever) 관련 하이퍼파라미터 설정
    st.subheader("검색 설정")
    k_value = st.slider(
        "검색할 문서 조각 개수 (K)",
        min_value=1,
        max_value=10,
        value=4,
        step=1
    )

    # 3-3. 신규 지식 베이스(문서) 업로드 파이프라인
    st.subheader("문서 관리")
    uploaded_files = st.file_uploader(
        "매뉴얼 파일 업로드 (PDF, DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"총 {len(uploaded_files)}개의 파일이 선택되었습니다.")
        if st.button("문서 DB화 시작"):
            with st.spinner("문서를 파싱하고 벡터 DB에 적재 중입니다..."):
                time.sleep(2)  # TODO: 실제 Ingestion 파이프라인 연동 필요
                st.success("문서 DB 적재가 완료되었습니다.")

    st.markdown("---")

    # 3-4. 시스템 상태 및 벡터 DB 초기화 제어
    st.subheader("시스템 관리")
    if st.button("벡터 DB 초기화", help="저장된 모든 임베딩 데이터를 영구 삭제합니다."):
        st.warning("초기화 프로세스가 실행되었습니다.")
        st.session_state.messages = []  # 컨텍스트 초기화 처리

# ==============================================================================
# 4. 메인 채팅 인터페이스 렌더링
# ==============================================================================
st.title("기업 매뉴얼 기반 지능형 챗봇")
st.markdown(f"현재 모델: **{selected_model}** | 검색 개수(K): **{k_value}**")
st.markdown("---")

# 누적된 대화 히스토리를 UI에 순차적으로 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================================================================
# 5. 사용자 질의 입력 및 RAG 파이프라인 실행 로직
# ==============================================================================
if prompt := st.chat_input("궁금한 점을 입력해 주세요."):
    # 5-1. 사용자 질의 출력 및 세션 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5-2. 시스템 응답 생성 및 스트리밍 시뮬레이션
    # IDE 경고 해결: 중첩된 컨텍스트 매니저(with) 결합
    with st.chat_message("assistant"), st.spinner("답변을 생성하는 중입니다..."):
        time.sleep(1)  # TODO: 실제 RAG Chain invoke/stream 연동 시 제거

        # RAG 연동 전 UI 테스트용 Mock 응답 데이터
        response = (
            f"'{prompt}'에 대한 분석 결과입니다. (현재 백엔드 연동 대기 중)\n\n"
            "**참조 출처:** [운영매뉴얼.pdf, 12p]"
        )
        st.markdown(response)

    # 5-3. 어시스턴트 응답 세션 누적 처리
    st.session_state.messages.append({"role": "assistant", "content": response})
