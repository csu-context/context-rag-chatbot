from src.utils.paths import ensure_directories

def main():
    """
    RAG 챗봇 프로젝트의 메인 진입점.
    프로젝트 실행에 필요한 초기화 작업을 수행합니다.
    """
    print("RAG 챗봇 초기화 중...")
    
    # 1. 필수 디렉토리 확인 및 생성
    ensure_directories()
    
    print("초기화 완료. 메인 로직을 시작할 준비가 되었습니다.")
    # TODO: 챗봇 엔진 또는 서버 실행 로직 추가

if __name__ == "__main__":
    main()
