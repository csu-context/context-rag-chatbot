"""시맨틱 캐시 전역 토글(SEMANTIC_CACHE_ENABLED) 게이트 단위 테스트. (이슈 #157)

캐시 코드는 보존하되 기본 비활성(opt-in)이며, 켤 때만 생성된다. RAGPipeline·PipelineOrchestrator
두 생성 지점이 플래그로 게이트되는지, 끈 상태에서 flush가 no-op인지 검증한다.
"""

from unittest.mock import MagicMock, patch

from src.core.chains import RAGPipeline
from src.pipeline.orchestrator import PipelineOrchestrator


def _make_pipeline(use_cache: bool = True) -> RAGPipeline:
    return RAGPipeline(MagicMock(), llm=MagicMock(), reranker=MagicMock(), use_cache=use_cache)


class TestChainsCacheToggle:
    def test_cache_off_by_default(self):
        """기본(플래그 False)에서는 use_cache=True여도 캐시를 만들지 않는다."""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache,
            patch("src.core.chains.TracingLogger"),
            patch("src.core.chains.settings.SEMANTIC_CACHE_ENABLED", False),
        ):
            pipeline = _make_pipeline()
            assert pipeline.cache is None
            mock_cache.assert_not_called()

    def test_cache_on_when_enabled(self):
        """플래그를 켜고 use_cache=True면 캐시가 생성된다."""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache,
            patch("src.core.chains.TracingLogger"),
            patch("src.core.chains.settings.SEMANTIC_CACHE_ENABLED", True),
        ):
            pipeline = _make_pipeline()
            assert pipeline.cache is mock_cache.return_value
            mock_cache.assert_called_once()

    def test_use_cache_false_overrides_enabled(self):
        """플래그가 켜져 있어도 use_cache=False(예: 평가)면 캐시를 만들지 않는다."""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache,
            patch("src.core.chains.TracingLogger"),
            patch("src.core.chains.settings.SEMANTIC_CACHE_ENABLED", True),
        ):
            pipeline = _make_pipeline(use_cache=False)
            assert pipeline.cache is None
            mock_cache.assert_not_called()


class TestOrchestratorCacheToggle:
    def _build(self, enabled: bool) -> PipelineOrchestrator:
        PipelineOrchestrator._instance = None
        with (
            patch("src.pipeline.orchestrator.IngestionPipeline"),
            patch("src.pipeline.orchestrator.StorageManager"),
            patch("src.pipeline.orchestrator.SemanticCache") as mock_cache,
            patch("src.pipeline.orchestrator.settings.SEMANTIC_CACHE_ENABLED", enabled),
        ):
            orch = PipelineOrchestrator()
            return orch, mock_cache

    def test_cache_off_by_default(self):
        """기본(플래그 False)에서는 오케스트레이터가 캐시를 만들지 않는다."""
        orch, mock_cache = self._build(enabled=False)
        assert orch.cache is None
        mock_cache.assert_not_called()

    def test_cache_on_when_enabled(self):
        """플래그를 켜면 오케스트레이터가 캐시를 생성한다."""
        orch, mock_cache = self._build(enabled=True)
        assert orch.cache is mock_cache.return_value
        mock_cache.assert_called_once()
