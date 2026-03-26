# Context: RAG-based AI Q\&A Chatbot

> **조선대학교 AI·SW학부(컴퓨터공학전공) 산학 프로젝트1 - 기업 내부 매뉴얼 기반 AI 질의응답 시스템**

본 프로젝트는 기업의 내부 매뉴얼(PDF, Docx 등)을 효율적으로 분석하고, 사용자의 질문에 정확한 근거를 바탕으로 답변하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

-----

## 프로젝트 환경 표준화

팀원 간 개발 환경 차이로 인한 오류를 방지하기 위해 아래 설정을 준수합니다.

  * **Python 버전:** 3.13.9
  * **인코딩:** UTF-8
  * **패키지 관리:** `requirements.txt`
  * **환경 변수:** `.env` (보안을 위해 `.env.example` 템플릿 사용)

-----

## Git 설정 (Line Ending)

윈도우와 맥의 줄 바꿈 처리 방식이 다르므로, 프로젝트 시작 전 터미널에서 아래 명령어를 반드시 입력해 주세요.

  * **Windows 사용자:**
    ```bash
    git config --global core.autocrlf true
    ```
  * **macOS 사용자:**
    ```bash
    git config --global core.autocrlf input
    ```

-----

## 시작하기 (Setup Guide)

팀원들은 파이참(PyCharm)을 활용해 아래 순서대로 로컬 환경을 세팅해 주세요.

### 1\. 프로젝트 가져오기 (Clone)

1.  파이참 실행 후 **[Get from VCS]** 클릭
2.  URL 칸에 레포지토리 주소 입력 후 **[Clone]**
      * `[https://github.com/csu-context/context-rag-chatbot.git](https://github.com/csu-context/context-rag-chatbot.git)`

### 2\. 가상환경 및 인터프리터 설정

1.  프로젝트가 열리면 우측 하단 팝업에서 **[Create Virtual Environment]** 클릭
2. 자동 팝업이 뜨지 않는 경우:
   * `Settings` (또는 `Cmd + ,`) > `Project: context-rag-chatbot` > `Python Interpreter`
   * `Add Interpreter` > `Add Local Interpreter` > `Virtualenv Environment` 선택 후 **OK**

### 3. 패키지 설치
1. 에디터 상단에 나타나는 **"Package requirements 'requirements.txt' are not satisfied"** 노란색 바 확인
2. **[Install requirements]** 링크를 클릭하여 모든 라이브러리 설치

### 4. 환경 변수 설정
1. 프로젝트 루트에 `.env` 파일을 새로 생성
2. `.env.example`의 내용을 복사하여 붙여넣고, 본인의 API 키를 입력 (키는 팀장에게 문의)

---

## 협업 규칙 (Ground Rules)

1. **브랜치 전략:** 직접 `main`에 푸시하지 않습니다.
    * `feature/기능명-이름` 브랜치에서 작업 후 PR(Pull Request)을 생성합니다.
2. **경로 관리:** 운영체제 간 호환성을 위해 **pathlib** 라이브러리를 사용합니다.
    * **Good:** `Path("data") / "manual.pdf"`
    * **Bad:** `"data\\manual.pdf"`
3. **커밋 메시지:** 접두어를 사용하여 명확하게 작성합니다.
    * `feat:` 새로운 기능 추가 / `fix:` 버그 수정 / `docs:` 문서 수정
