import subprocess
import sys

# Windows에서 Python 모듈로 실행하기 위해 sys.executable 사용
result = subprocess.run([sys.executable, "-m", "pytest", "src/tests/integration/test_reranker.py", "-k", "test_timeout_dynamic_top_k", "-v"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
sys.exit(result.returncode)
