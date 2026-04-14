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

---

## 개발 프로세스 (Workflow)

팀원 모두의 코드 품질과 히스토리 관리를 위해 아래의 **GitHub Flow** 과정을 엄격히 준수합니다.



### 1. 이슈(Issue) 생성
*   새로운 기능 구현이나 버그 수정 시작 전, **[Issues]** 탭에서 이슈를 먼저 생성합니다.
*   이슈 템플릿에 따라 작업 내용을 간략히 적고, **Assignees**와 **Labels**를 설정합니다.

### 2. 브랜치 생성 (Branching)
*   모든 작업 브랜치는 항상 `develop` 브랜치로부터 분기합니다.
*   **브랜치 이름 규칙:** `type/기능명-이름` (예: `feature/login`)
*   터미널 명령어 예시:
    ```bash
    git checkout develop
    git pull origin develop
    git checkout -b feature/issue-number-name
    ```

### 3. 개발 및 커밋 (Commit)
*   작업 단위별로 커밋을 남기며, 커밋 메시지 컨벤션을 준수합니다.
*   가능한 한 하나의 커밋에는 하나의 작업 내용만 담습니다.

### 4. 풀 리퀘스트(PR) 및 코드 리뷰
*   개발이 완료되면 `develop` 브랜치를 대상으로 **Pull Request**를 생성합니다.
*   **Issue 연동:** PR 설명란에 `Closes #이슈번호`를 기재하여 관련 이슈가 자동으로 닫히도록 합니다.
*   최소 1명 이상의 팀원에게 **Approve(승인)**를 받은 후 `develop`에 병합(Merge)합니다.

---

## 코드 스타일 및 자동화 (Linting & Formatting)

본 프로젝트는 일관된 코드 스타일 유지와 품질 관리를 위해 **Ruff**와 **Black**을 사용합니다. 모든 팀원은 커밋 전 아래 명령어를 통해 코드를 정돈해 주세요.

### 1. 도구 설치
`requirements.txt`를 통해 패키지를 설치하면 자동으로 포함됩니다.
```bash
pip install -r requirements.txt
```

### 2. 코드 검사 및 자동 수정 (Ruff)
문법 오류, 미사용 임포트 등을 검사하고 가능한 경우 자동으로 수정합니다.
```bash
ruff check . --fix
```

### 3. 코드 포맷팅 (Black)
코드의 공백, 줄바꿈 등 외형을 일관되게 교정합니다.
```bash
black .
```

> **참고:** GitHub Actions가 설정되어 있어, **Pull Request 생성 시 자동으로 린트 및 포맷팅 검사가 수행**됩니다. 검사를 통과하지 못하면 병합(Merge)이 불가능할 수 있으니 로컬에서 먼저 확인해 주세요.

---

## 협업 금지 사항 (Strictly Prohibited)

현재 Private 레포지토리 특성상 시스템적인 제약이 없으므로, 아래 사항은 **절대 금지**하며 팀원 간 신뢰를 바탕으로 준수합니다.

1.  **`main` 브랜치 직접 푸시 및 병합 금지**
    *   `main`은 배포용 브랜치입니다. 모든 코드는 `develop`을 거쳐 검증된 후 팀장 주도하에 병합합니다.
2.  **`develop` 브랜치 직접 푸시 금지**
    *   사소한 오타 수정이라도 반드시 브랜치를 생성하고 PR 과정을 거쳐야 합니다.
3.  **코드 리뷰 없는 독단적 병합 금지**
    *   팀원의 확인을 받지 않은 코드는 코드 베이스에 혼란을 줄 수 있습니다. 반드시 승인 후 병합하세요.
4.  **이슈 생성 없는 개발 진행 금지**
    *   "누가 어떤 작업을 하고 있는지" 기록을 남기는 것이 협업의 시작입니다. 무조건 이슈 먼저!

