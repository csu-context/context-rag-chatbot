from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.orchestrator import PipelineOrchestrator


class PreprocessingPipeline:
    """하위 호환성을 위한 기존 PreprocessingPipeline 클래스 래퍼"""

    def __init__(self, *args, **kwargs):
        self.orchestrator = PipelineOrchestrator()

    def run(self, *args, **kwargs):
        return self.orchestrator.run_ingestion()


__all__ = ["IngestionPipeline", "PipelineOrchestrator", "PreprocessingPipeline"]
