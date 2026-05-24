from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import PipelineOrchestrator


class TestParserSelection:
    @pytest.fixture
    def orchestrator(self):
        """PipelineOrchestrator 싱글톤을 초기화하고 테스트를 위한 모의 객체들을 연동합니다."""
        # 싱글톤 인스턴스 초기화 우회 또는 재설정을 위해 _instance를 None으로 강제 설정
        PipelineOrchestrator._instance = None

        with (
            patch("src.pipeline.ChromaDBManager") as mock_chroma_class,
            patch("src.pipeline.SemanticCache") as mock_cache_class,
            patch("src.pipeline.settings") as mock_settings,
        ):
            mock_chroma_class.return_value = MagicMock()
            mock_cache_class.return_value = MagicMock()
            mock_settings.PARSER_TYPE = "manual"

            orchestrator = PipelineOrchestrator()
            yield orchestrator

    def test_parser_type_dynamic_switch(self, orchestrator):
        """run_ingestion 호출 시 parser_type 파라미터에 따라 파서 전략이 동적으로 전환되는지 검증합니다."""
        # 초기 상태 확인 (기본값 settings.PARSER_TYPE = "manual"로 로드됨)
        assert orchestrator.parser_type == "manual"

        # 헬퍼 및 서브 모듈 모킹
        orchestrator.ingestion_pipeline = MagicMock()
        orchestrator.ingestion_pipeline.scan_files.return_value = []
        orchestrator._load_manifest = MagicMock(return_value={})

        # docling 라이브러리가 로드되는 상황을 모킹하기 위해 patch 사용
        with patch("src.pipeline.DoclingPDFParserStrategy") as mock_docling_strategy:
            mock_docling_strategy.return_value = MagicMock()

            # docling으로 전환 호출
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: MagicMock() if name == "docling" else None,
            ):
                orchestrator.run_ingestion(parser_type="docling")

                # 검증
                assert orchestrator.parser_type == "docling"

    def test_progress_callback_execution(self, orchestrator):
        """동기화 파이프라인 가동 시 progress_callback이 정상적으로 호출되는지 검증합니다."""
        orchestrator.ingestion_pipeline = MagicMock()

        # 3개의 모의 파일 스캔 결과
        mock_files = [Path("file1.pdf"), Path("file2.pdf"), Path("file3.pdf")]
        orchestrator.ingestion_pipeline.scan_files.return_value = mock_files

        # manifest와 delta 계산 모킹하여 3개 파일 모두 변경된 것으로 설정
        orchestrator._load_manifest = MagicMock(return_value={})
        orchestrator._calculate_delta = MagicMock(return_value=(mock_files, [], {}))

        callback_calls = []

        def dummy_callback(current, total, file_name):
            callback_calls.append((current, total, file_name))

        # process_and_chunk가 호출될 때 콜백을 직접 구동하도록 모의 구현
        def mock_process_and_chunk(files, progress_callback=None):
            for idx, f in enumerate(files):
                if progress_callback:
                    progress_callback(idx, len(files), f.name)
            if progress_callback:
                progress_callback(len(files), len(files), "완료")
            return []

        orchestrator.ingestion_pipeline.process_and_chunk = mock_process_and_chunk
        orchestrator.ingestion_pipeline.save_processed_data = MagicMock(return_value=[])
        orchestrator.ingestion_pipeline.upsert_to_db = MagicMock()

        # 실행
        orchestrator.run_ingestion(parser_type="manual", progress_callback=dummy_callback)

        # callback_calls에 기록된 호출 횟수 및 인자 검증
        # 3개 파일에 대해 파일 단위 호출 + 최종 완료 호출 = 4번 호출되어야 함
        assert len(callback_calls) == 4
        assert callback_calls[0] == (0, 3, "file1.pdf")
        assert callback_calls[1] == (1, 3, "file2.pdf")
        assert callback_calls[2] == (2, 3, "file3.pdf")
        assert callback_calls[3] == (3, 3, "완료")
