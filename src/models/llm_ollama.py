import contextlib
import logging
import os
import threading
import time
from typing import Any, ClassVar

# noinspection PyPackageRequirements
import requests
from langchain_ollama import ChatOllama

from src.models.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OllamaPullStatus:
    """
    Ollama 모델 다운로드의 백그라운드 진행 상태를 관리하는 스레드 안전(Thread-safe) 클래스입니다.
    """

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _instances: ClassVar[dict] = {}  # model_name -> status_info dict

    @classmethod
    def get_status(cls, model_name: str) -> dict | None:
        """지정된 모델의 다운로드 상태를 반환합니다."""
        with cls._lock:
            status_info = cls._instances.get(model_name)
            if status_info:
                return {
                    "status": status_info["status"],
                    "completed": status_info["completed"],
                    "total": status_info["total"],
                    "message": status_info["message"],
                }
            return None

    @classmethod
    def _run_pull(cls, model: "OllamaModel", status_info: dict) -> None:
        """백그라운드 모델 다운로드 작업을 수행하는 내부 메서드입니다."""
        # noinspection PyBroadException
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
                    if status in ("success", "error"):
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
            with contextlib.suppress(Exception):
                if model.is_model_available():
                    with cls._lock:
                        status_info["status"] = "success"

    @classmethod
    def start_pull(cls, model: "OllamaModel") -> None:
        """모델 다운로드를 백그라운드 스레드에서 시작합니다."""
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
                "thread": None,
            }
            cls._instances[model_name] = status_info

            thread = threading.Thread(target=cls._run_pull, args=(model, status_info), daemon=True)
            status_info["thread"] = thread
            thread.start()


class OllamaModel(BaseLLM):
    """
    Ollama를 통해 로컬 환경에서 구동되는 sLLM(Small Large Language Model)을 제어하는 인스턴스 클래스입니다.
    """

    def __init__(
        self, model_name: str | None = None, base_url: str | None = None, temperature: float = 0.1, **kwargs: Any
    ):
        """
        Ollama 모델 초기화 및 헬스 체크 수행

        :param model_name: 구동할 Ollama 모델 명 (기본값: 환경변수 MODEL_NAME 또는 'phi3')
        :param base_url: Ollama 서버 주소 (기본값: 환경변수 OLLAMA_BASE_URL 또는 'http://localhost:11434')
        :param temperature: 텍스트 생성 무작위성 제어 (기본값: 0.1)
        :param kwargs: LangChain ChatOllama에 전달될 추가 파라미터 (num_ctx 등)
        """
        if not model_name:
            model_name = os.getenv("MODEL_NAME", "phi3")

        super().__init__(model_name=model_name)
        self._model_name = model_name

        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not self.base_url:
            raise ValueError("Ollama 서버 주소(base_url) 설정이 올바르지 않습니다.")

        num_ctx = kwargs.pop("num_ctx", 2048)
        num_predict = kwargs.pop("num_predict", 512)
        repeat_penalty = kwargs.pop("repeat_penalty", 1.2)

        self.llm = ChatOllama(
            model=self._model_name,
            base_url=self.base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            repeat_penalty=repeat_penalty,
            **kwargs,
        )

        if not self.check_health():
            logger.warning(
                f"Ollama 서비스 헬스체크 실패 ({self.base_url}). "
                "서버 구동 상태를 확인하십시오. 서비스 미가동 시 RAG 응답 생성이 불가합니다."
            )

    def check_health(self) -> bool:
        """Ollama API 엔드포인트의 가용성을 확인합니다."""
        import urllib.request

        # noinspection PyBroadException
        try:
            url = self.base_url.rstrip("/") + "/api/tags"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                return response.status == 200
        except Exception:
            return False

    def is_model_available(self) -> bool:
        """대상 모델이 로컬 Ollama 환경에 다운로드되어 있는지 검증합니다."""
        import json
        import urllib.request

        # noinspection PyBroadException
        try:
            url = self.base_url.rstrip("/") + "/api/tags"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status != 200:
                    return False
                data = json.loads(response.read().decode("utf-8"))
                models = data.get("models", [])

                target = self._model_name
                for m in models:
                    name = m.get("name", "")
                    if name == target or name == f"{target}:latest" or (":" in name and name.split(":")[0] == target):
                        return True
                    if name.startswith(target + ":") or target.startswith(name + ":"):
                        return True
                return False
        except Exception:
            return False

    def pull_model_progress(self) -> Any:
        """Ollama API를 통해 모델을 다운로드하고 진행 상태 스트림을 반환합니다."""
        import json

        url = self.base_url.rstrip("/") + "/api/pull"
        payload = {"name": self._model_name, "stream": True}

        try:
            response = requests.post(url, json=payload, stream=True, timeout=(5.0, 30.0))
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8").strip()
                    with contextlib.suppress(json.JSONDecodeError):
                        yield json.loads(line_str)
        except requests.exceptions.RequestException as e:
            yield {"status": "error", "message": f"Ollama 연결 실패: {e!s}"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """동기 방식으로 Ollama 모델을 호출하여 텍스트를 생성합니다."""
        start_time = time.time()
        # noinspection PyBroadException
        try:
            response = self.llm.invoke(prompt, **kwargs)
            latency = time.time() - start_time

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
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
                model_name=self._model_name,
                metadata=getattr(response, "response_metadata", {}),
                cost=0.0,
            )
        except Exception as e:
            if not self.check_health():
                msg = (
                    f"Ollama 서비스({self.base_url})에 연결할 수 없습니다. "
                    "서버 활성화 및 대상 모델 다운로드 상태를 확인하십시오."
                )
                logger.error(msg)
                raise ConnectionError(msg) from e
            logger.error(f"Ollama({self._model_name}) 동기 호출 중 오류 발생: {e}")
            raise e

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """비동기 방식으로 Ollama 모델을 호출하여 텍스트를 생성합니다."""
        start_time = time.time()
        # noinspection PyBroadException
        try:
            response = await self.llm.ainvoke(prompt, **kwargs)
            latency = time.time() - start_time

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
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
                model_name=self._model_name,
                metadata=getattr(response, "response_metadata", {}),
                cost=0.0,
            )
        except Exception as e:
            logger.error(f"Ollama({self._model_name}) 비동기 호출 중 오류 발생: {e}")
            raise e

    def get_model(self) -> ChatOllama:
        """초기화된 LangChain ChatOllama 인스턴스를 반환합니다."""
        return self.llm

    def start_pull_background(self) -> None:
        """백그라운드에서 비동기적으로 모델 다운로드를 실행합니다."""
        OllamaPullStatus.start_pull(self)

    def get_pull_status(self) -> dict | None:
        """현재 백그라운드 모델 다운로드 진행 상태를 조회합니다."""
        return OllamaPullStatus.get_status(self._model_name)

    @property
    def model_name(self) -> str:
        """현재 구동 중인 모델 명을 반환합니다."""
        return self._model_name

    @property
    def model_type(self) -> str:
        """모델의 플랫폼 타입을 반환합니다."""
        return "ollama"
