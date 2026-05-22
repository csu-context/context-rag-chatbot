import logging
import os
import threading
import time
from typing import Any

from langchain_ollama import ChatOllama

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OllamaPullStatus:
    """Ollama 모델 다운로드의 백그라운드 진행 상태를 관리하는 스레드 안전 클래스"""
    _lock = threading.Lock()
    _instances = {}  # model_name: {status: str, completed: int, total: int, message: str, thread: Thread}

    @classmethod
    def get_status(cls, model_name: str) -> dict | None:
        with cls._lock:
            status_info = cls._instances.get(model_name)
            if status_info:
                return {
                    "status": status_info["status"],
                    "completed": status_info["completed"],
                    "total": status_info["total"],
                    "message": status_info["message"]
                }
            return None

    @classmethod
    def start_pull(cls, model: "OllamaModel") -> None:
        model_name = model.model_name
        with cls._lock:
            if model_name in cls._instances:
                status_info = cls._instances[model_name]
                thread = status_info.get("thread")
                if thread and thread.is_alive():
                    return

            status_info = {
                "status": "pulling",
                "completed": 0,
                "total": 0,
                "message": "",
                "thread": None
            }
            cls._instances[model_name] = status_info

            def run_pull():
                try:
                    if model.is_model_available():
                        with cls._lock:
                            status_info["status"] = "success"
                        return

                    last_check_time = time.time()
                    for progress in model.pull_model_progress():
                        status = progress.get("status", "")
                        with cls._lock:
                            status_info["status"] = status
                            status_info["completed"] = progress.get("completed", 0)
                            status_info["total"] = progress.get("total", 0)
                            status_info["message"] = progress.get("message", "")
                            if status == "success":
                                break
                            elif status == "error":
                                break

                        if time.time() - last_check_time > 5.0:
                            if model.is_model_available():
                                with cls._lock:
                                    status_info["status"] = "success"
                                break
                            last_check_time = time.time()
                except Exception as e:
                    with cls._lock:
                        status_info["status"] = "error"
                        status_info["message"] = str(e)
                finally:
                    try:
                        if model.is_model_available():
                            with cls._lock:
                                status_info["status"] = "success"
                    except Exception:
                        pass

            thread = threading.Thread(target=run_pull, daemon=True)
            status_info["thread"] = thread
            thread.start()


class OllamaModel(BaseLLM):
    """Ollama를 통한 로컬 sLLM 구현체"""

    def __init__(self, model_name: str, base_url: str | None = None, temperature: float = 0.1):
        """
        Args:
            model_name: Ollama 모델 명 (예: llama3, solar)
            base_url: Ollama 서버 주소 (기본값: http://localhost:11434)
            temperature: 생성 온도
        """
        super().__init__(model_name=model_name)
        # 환경 변수 또는 직접 전달된 URL 사용
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.model = ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
            num_predict=512,
            repeat_penalty=1.2,
        )
        # 헬스 체크 연동 경고 로그
        if not self.check_health():
            logger.warning(
                f"Ollama 서비스 헬스체크 실패 ({self.base_url}). "
                "로컬 sLLM(Ollama)이 구동 중인지 확인해 주세요. "
                "Ollama가 실행되지 않으면 RAG 응답을 생성할 수 없습니다."
            )

    def check_health(self) -> bool:
        """Ollama 서비스 구동 상태를 확인합니다."""
        import urllib.request

        try:
            url = self.base_url.rstrip("/") + "/api/tags"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                return response.status == 200
        except Exception:
            return False

    def is_model_available(self) -> bool:
        """Ollama 서비스에 대상 모델이 다운로드 완료되었는지 확인합니다."""
        import json
        import urllib.request

        try:
            url = self.base_url.rstrip("/") + "/api/tags"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status != 200:
                    return False
                data = json.loads(response.read().decode("utf-8"))
                models = data.get("models", [])

                target = self.model_name
                for m in models:
                    name = m.get("name", "")
                    if name == target:
                        return True
                    if ":" not in target and name == f"{target}:latest":
                        return True
                    if ":" in name and name.split(":")[0] == target:
                        return True
                    if name.startswith(target + ":") or target.startswith(name + ":"):
                        return True
                return False
        except Exception:
            return False

    def pull_model_progress(self) -> Any:
        """Ollama 서비스에서 모델을 다운로드하며 실시간 진행 상황을 생성합니다.

        Yields:
            dict: 진행 정보 딕셔너리
        """
        import json
        import requests

        url = self.base_url.rstrip("/") + "/api/pull"
        payload = {"name": self.model_name, "stream": True}

        try:
            response = requests.post(url, json=payload, stream=True, timeout=(5.0, 30.0))
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8").strip()
                    try:
                        yield json.loads(line_str)
                    except json.JSONDecodeError:
                        pass
        except requests.exceptions.RequestException as e:
            yield {"status": "error", "message": f"Ollama 연결 실패: {str(e)}"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def invoke(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        start_time = time.time()

        try:
            response = self.model.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            # Ollama는 로컬 구동이므로 명시적으로 비용을 0.0으로 설정
            usage = {}
            if hasattr(response, "usage_metadata"):
                meta = response.usage_metadata
                usage = {
                    "input_tokens": meta.get("input_token_count") or meta.get("input_tokens") or 0,
                    "output_tokens": meta.get("output_token_count") or meta.get("output_tokens") or 0,
                    "total_tokens": meta.get("total_token_count") or meta.get("total_tokens") or 0,
                }

            return LLMResponse(
                content=response.content,
                usage=usage,
                latency=latency,
                model_name=self.model_name,
                metadata=getattr(response, "response_metadata", {}),
                cost=0.0,  # 로컬 모델 비용 0원
            )

        except Exception as e:
            if not self.check_health():
                msg = (
                    f"Ollama 서비스({self.base_url})에 연결할 수 없습니다. "
                    "Ollama 서버가 활성화되어 있고 로컬 모델이 다운로드되어 있는지 확인하십시오."
                )
                logger.error(msg)
                raise ConnectionError(msg) from e
            logger.error(f"Ollama({self.model_name}) 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatOllama:
        return self.model

    def start_pull_background(self) -> None:
        """백그라운드에서 모델 다운로드를 실행합니다."""
        OllamaPullStatus.start_pull(self)

    def get_pull_status(self) -> dict | None:
        """현재 백그라운드 다운로드 상태를 조회합니다."""
        return OllamaPullStatus.get_status(self.model_name)
