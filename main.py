from pathlib import Path


# 메인 실행 함수
def main():
    # 운영체제에 상관없이 경로를 안전하게 생성함
    # Path("폴더명") / "파일명" 형식을 사용함
    data_path = Path("data") / "manual.pdf"

    print(f"현재 설정된 파일 경로는: {data_path}")


# 프로그램 시작점
if __name__ == '__main__':
    main()