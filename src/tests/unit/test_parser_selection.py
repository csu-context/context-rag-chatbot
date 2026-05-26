from unittest.mock import MagicMock, mock_open, patch

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
        from src.utils.paths import RAW_DATA_DIR

        orchestrator.ingestion_pipeline = MagicMock()

        # 절대 경로로 모킹 파일 제공
        mock_files = [RAW_DATA_DIR / "file1.pdf", RAW_DATA_DIR / "file2.pdf", RAW_DATA_DIR / "file3.pdf"]
        orchestrator.ingestion_pipeline.scan_files.return_value = mock_files

        # manifest와 delta 계산 모킹
        orchestrator._load_manifest = MagicMock(
            return_value={"version": "2.0", "global_parser_type": "manual", "files": {}}
        )
        orchestrator._calculate_delta = MagicMock(
            return_value=(mock_files, [], {"version": "2.0", "global_parser_type": "manual", "files": {}})
        )

        callback_calls = []

        def dummy_callback(current, total, file_name):
            callback_calls.append((current, total, file_name))

        # process_and_chunk가 호출될 때 콜백을 직접 구동하도록 모의 구현
        def mock_process_and_chunk(files, progress_callback=None, file_parser_types=None):
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
        assert len(callback_calls) == 4
        assert callback_calls[0] == (0, 3, "file1.pdf")
        assert callback_calls[1] == (1, 3, "file2.pdf")
        assert callback_calls[2] == (2, 3, "file3.pdf")
        assert callback_calls[3] == (3, 3, "완료")

    def test_manifest_backward_compatibility(self, orchestrator):
        """이전 규격(1.0)의 매니페스트를 로드할 때 2.0 포맷으로 자동 래핑 변환하는지 검증합니다."""
        mock_data = {"rules.pdf": "hash123", "info.md": "hash456"}
        import json

        mock_json = json.dumps(mock_data)

        with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", mock_open(read_data=mock_json)):
            loaded = orchestrator._load_manifest()

            # 검증: 2.0 구조로 자동 래핑 전환 여부
            assert loaded["version"] == "2.0"
            assert loaded["global_parser_type"] == "manual"
            assert "rules.pdf" in loaded["files"]
            assert loaded["files"]["rules.pdf"]["hash"] == "hash123"
            assert loaded["files"]["rules.pdf"]["parser_type"] == "manual"

    def test_calculate_delta_with_per_file_parser(self, orchestrator):
        """각 파일별로 다른 파서 타입이 지정되어 있을 때,
        해당 파서 규칙에 따라 개별적으로 델타 해시가 연산되는지 검증합니다.
        """
        from src.utils.paths import RAW_DATA_DIR

        mock_files = [RAW_DATA_DIR / "rules.pdf", RAW_DATA_DIR / "info.md"]

        # 2.0 규격 매니페스트
        old_manifest = {
            "version": "2.0",
            "global_parser_type": "manual",
            "files": {
                "rules.pdf": {"hash": "old_rules_hash", "parser_type": "docling"},
                "info.md": {"hash": "old_info_hash", "parser_type": "manual"},
            },
        }

        # generate_file_hash 함수가 호출될 때, 각 파서 타입이 의도대로 넘어가는지 모킹
        with patch("src.pipeline.generate_file_hash") as mock_hash_gen:
            mock_hash_gen.side_effect = lambda f, parser_type: f"hash_of_{f.name}_by_{parser_type}"

            _, _, new_manifest = orchestrator._calculate_delta(mock_files, old_manifest)

            # rules.pdf는 docling 기준, info.md는 manual 기준으로 해시가 호출되었는지 검증
            mock_hash_gen.assert_any_call(RAW_DATA_DIR / "rules.pdf", "docling")
            mock_hash_gen.assert_any_call(RAW_DATA_DIR / "info.md", "manual")

            # 생성된 new_manifest 값 검증
            files = new_manifest["files"]
            assert files["rules.pdf"]["hash"] == "hash_of_rules.pdf_by_docling"
            assert files["rules.pdf"]["parser_type"] == "docling"
            assert files["info.md"]["hash"] == "hash_of_info.md_by_manual"
            assert files["info.md"]["parser_type"] == "manual"

    def test_update_file_parser_execution(self, orchestrator):
        """특정 파일의 파서 변경 함수 호출 시, 매니페스트에 정상 반영되고 인덱싱이 기동되는지 검증합니다."""
        old_manifest = {
            "version": "2.0",
            "global_parser_type": "manual",
            "files": {"rules.pdf": {"hash": "hash_val", "parser_type": "manual"}},
        }

        orchestrator._load_manifest = MagicMock(return_value=old_manifest)
        orchestrator._save_manifest = MagicMock()
        orchestrator.run_ingestion = MagicMock()

        # rules.pdf의 파서를 docling으로 변경 실행
        orchestrator.update_file_parser("rules.pdf", "docling")

        # 매니페스트 갱신 저장 확인
        orchestrator._save_manifest.assert_called_once()
        saved_manifest = orchestrator._save_manifest.call_args[0][0]
        assert saved_manifest["files"]["rules.pdf"]["parser_type"] == "docling"

        # 동기화 프로세스 실행 확인 (target_files 리스트 전달 검증)
        orchestrator.run_ingestion.assert_called_once()
        call_kwargs = orchestrator.run_ingestion.call_args[1]
        assert call_kwargs["force"] is False
        assert call_kwargs["progress_callback"] is None
        assert len(call_kwargs["target_files"]) == 1
        assert call_kwargs["target_files"][0].name == "rules.pdf"

    def test_update_multiple_file_parsers_execution(self, orchestrator):
        """복수 파일의 파서 변경 함수 호출 시, 매니페스트에 일괄 반영되고 인덱싱이 기동되는지 검증합니다."""
        old_manifest = {
            "version": "2.0",
            "global_parser_type": "manual",
            "files": {
                "rules.pdf": {"hash": "hash1", "parser_type": "manual"},
                "info.md": {"hash": "hash2", "parser_type": "manual"},
            },
        }

        orchestrator._load_manifest = MagicMock(return_value=old_manifest)
        orchestrator._save_manifest = MagicMock()
        orchestrator.run_ingestion = MagicMock()

        # rules.pdf와 info.md를 모두 docling으로 변경 실행
        orchestrator.update_multiple_file_parsers({"rules.pdf": "docling", "info.md": "docling"})

        # 매니페스트 갱신 저장 확인
        orchestrator._save_manifest.assert_called_once()
        saved_manifest = orchestrator._save_manifest.call_args[0][0]
        assert saved_manifest["files"]["rules.pdf"]["parser_type"] == "docling"
        assert saved_manifest["files"]["info.md"]["parser_type"] == "docling"

        # 동기화 프로세스 실행 확인 (target_files 리스트 전달 검증)
        orchestrator.run_ingestion.assert_called_once()
        call_kwargs = orchestrator.run_ingestion.call_args[1]
        assert call_kwargs["force"] is False
        assert call_kwargs["progress_callback"] is None
        assert len(call_kwargs["target_files"]) == 2
        assert {p.name for p in call_kwargs["target_files"]} == {"rules.pdf", "info.md"}
