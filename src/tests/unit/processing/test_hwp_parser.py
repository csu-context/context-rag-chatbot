from unittest.mock import MagicMock, patch

import pytest

from src.processing.hwp_parser import HwpParser


class TestHwpParserGetConverter:
    def test_initializes_converter_on_first_call(self):
        """_get_converter 최초 호출 시 MarkItDown 인스턴스가 생성되는지 검증."""
        mock_md_cls = MagicMock()
        mock_md_instance = MagicMock()
        mock_md_cls.return_value = mock_md_instance

        with patch.dict("sys.modules", {"markitdown": MagicMock(MarkItDown=mock_md_cls)}):
            parser = HwpParser()
            assert parser._converter is None
            result = parser._get_converter()
            assert result == mock_md_instance
            assert parser._converter == mock_md_instance
            mock_md_cls.assert_called_once()

    def test_converter_cached_after_first_call(self):
        """두 번째 호출 시 동일 인스턴스를 재사용하는지 검증 (Lazy init 캐싱)."""
        mock_md_cls = MagicMock()
        mock_md_instance = MagicMock()
        mock_md_cls.return_value = mock_md_instance

        with patch.dict("sys.modules", {"markitdown": MagicMock(MarkItDown=mock_md_cls)}):
            parser = HwpParser()
            first = parser._get_converter()
            second = parser._get_converter()

        assert first is second
        mock_md_cls.assert_called_once()

    def test_raises_import_error_when_markitdown_missing(self):
        """markitdown 미설치 환경에서 ImportError가 적절한 메시지와 함께 발생하는지 검증."""
        parser = HwpParser()

        with (
            patch("builtins.__import__", side_effect=ImportError("No module named 'markitdown'")),
            pytest.raises(ImportError, match="markitdown-hwp"),
        ):
            parser._get_converter()


class TestHwpParserParse:
    def test_returns_markdown_and_extension_for_hwp(self, tmp_path):
        """HWP 파일 파싱 결과에 markdown과 extension 키가 포함되는지 검증."""
        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"dummy")

        mock_result = MagicMock()
        mock_result.text_content = "# 제1조\n\n내용입니다."

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        parser = HwpParser()
        parser._converter = mock_converter

        result = parser.parse(hwp_file)

        assert result["markdown"] == "# 제1조\n\n내용입니다."
        assert result["extension"] == "hwp"
        mock_converter.convert.assert_called_once_with(str(hwp_file))

    def test_returns_correct_extension_for_hwpx(self, tmp_path):
        """HWPX 파일의 extension이 'hwpx'로 반환되는지 검증."""
        hwpx_file = tmp_path / "문서.hwpx"
        hwpx_file.write_bytes(b"dummy")

        mock_result = MagicMock()
        mock_result.text_content = "내용"

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        parser = HwpParser()
        parser._converter = mock_converter

        result = parser.parse(hwpx_file)

        assert result["extension"] == "hwpx"

    def test_handles_none_text_content(self, tmp_path):
        """text_content가 None일 경우 빈 문자열로 처리되는지 검증."""
        hwp_file = tmp_path / "empty.hwp"
        hwp_file.write_bytes(b"dummy")

        mock_result = MagicMock()
        mock_result.text_content = None

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        parser = HwpParser()
        parser._converter = mock_converter

        result = parser.parse(hwp_file)

        assert result["markdown"] == ""

    def test_accepts_string_path(self, tmp_path):
        """파일 경로를 문자열로 전달해도 정상 동작하는지 검증."""
        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"dummy")

        mock_result = MagicMock()
        mock_result.text_content = "내용"

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        parser = HwpParser()
        parser._converter = mock_converter

        result = parser.parse(str(hwp_file))

        assert result["extension"] == "hwp"
        mock_converter.convert.assert_called_once_with(str(hwp_file))
