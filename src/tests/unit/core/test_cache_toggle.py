"""시맨틱 캐시 런타임 토글(SEMANTIC_CACHE_ENABLED) 게이트 단위 테스트. (이슈 #157)

캐시 코드는 보존하되 기본 비활성(opt-in)이고, @st.cache_resource로 파이프라인/오케스트레이터가
싱글톤 공유되므로 재기동 없이 런타임에 on/off되어야 한다. 따라서 생성자 시점이 아닌 호출 시점
(_get_cache)에 플래그를 판별하고, 켜진 경우에만 최초 1회 lazy 생성 후 메모이즈한다(기본 OFF면 idle 0).
"""

from unittest.mock import MagicMock, patch

from src.core.chains import RAGPipeline
from src.pipeline.orchestrator import PipelineOrchestrator


def _make_pipeline(use_cache: bool = True) -> RAGPipeline:
    return RAGPipeline(MagicMock(), llm=MagicMock(), reranker=MagicMock(), use_cache=use_cache)


class TestChainsCacheToggle:
    def test_cache_off_by_default(self):
        """기본(플래그 False)이면 호출해도 캐시를 만들지 않는다(lazy, idle 0)."""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache,
            patch("src.core.chains.TracingLogger"),
            patch("src.core.chains.settings.SEMANTIC_CACHE_ENABLED", False),
        ):
            pipeline = _make_pipeline()
            assert pipeline._get_cache() is None
            mock_cache.assert_not_called()

    def test_cache_lazy_built_when_enabled(self):
        """플래그 ON이면 첫 호출에 lazy 생성되고 이후 메모이즈된다(생성은 단 한 번)."""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache,
            patch("src.core.chains.TracingLogger"),
            patch("src.core.chains.settings.SEMANTIC_CACHE_ENABLED", True),
        ):
            pipeline = _make_pipeline()
            first = pipeline._get_cache()
            second = pipeline._get_cache()
            assert first is mock_cache.return_value
            assert first is second
            mock_cache.assert_called_once()

    def test_use_cache_false_overrides_enabled(self):
        """플래그가 켜져 있어도 use_cache=False(예: 평가)면 캐시를 만들지 않는다."""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache,
            patch("src.core.chains.TracingLogger"),
            patch("src.core.chains.settings.SEMANTIC_CACHE_ENABLED", True),
        ):
            pipeline = _make_pipeline(use_cache=False)
            assert pipeline._get_cache() is None
            mock_cache.assert_not_called()

    def test_runtime_toggle_without_rebuild(self):
        """싱글톤 인스턴스 재생성 없이 런타임 플래그 변화만으로 on/off되고, 생성은 1회뿐이다."""
        with (
            patch("src.core.chains.SemanticCache") as mock_cache,
            patch("src.core.chains.TracingLogger"),
            patch("src.core.chains.settings") as mock_settings,
        ):
            mock_settings.SEMANTIC_CACHE_ENABLED = False
            pipeline = _make_pipeline()
            assert pipeline._get_cache() is None  # OFF: 미생성

            mock_settings.SEMANTIC_CACHE_ENABLED = True
            assert pipeline._get_cache() is mock_cache.return_value  # 재기동 없이 ON

            mock_settings.SEMANTIC_CACHE_ENABLED = False
            assert pipeline._get_cache() is None  # 다시 OFF면 미반환

            mock_cache.assert_called_once()  # 생성은 단 한 번


class TestOrchestratorCacheToggle:
    def setup_method(self):
        PipelineOrchestrator._instance = None

    def teardown_method(self):
        PipelineOrchestrator._instance = None

    def _patches(self, enabled: bool):
        return (
            patch("src.pipeline.orchestrator.IngestionPipeline"),
            patch("src.pipeline.orchestrator.StorageManager"),
            patch("src.pipeline.orchestrator.SemanticCache"),
            patch("src.pipeline.orchestrator.settings.SEMANTIC_CACHE_ENABLED", enabled),
        )

    def test_cache_off_by_default(self):
        """기본(플래그 False)에서는 호출해도 캐시를 만들지 않는다."""
        ingestion, storage, mock_cache, flag = self._patches(enabled=False)
        with ingestion, storage, mock_cache as cache_cls, flag:
            orch = PipelineOrchestrator()
            assert orch._get_cache() is None
            cache_cls.assert_not_called()

    def test_cache_lazy_built_when_enabled(self):
        """플래그 ON이면 호출 시 lazy 생성되고 이후 메모이즈된다(생성은 단 한 번)."""
        ingestion, storage, mock_cache, flag = self._patches(enabled=True)
        with ingestion, storage, mock_cache as cache_cls, flag:
            orch = PipelineOrchestrator()
            first = orch._get_cache()
            second = orch._get_cache()
            assert first is cache_cls.return_value
            assert first is second
            cache_cls.assert_called_once()
