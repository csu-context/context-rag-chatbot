from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.strategies import HwpParserStrategy


@pytest.fixture()
def hwp_strategy():
    return HwpParserStrategy()


@pytest.fixture()
def mock_hwp_parser():
    """HwpParser.parse 결과를 모킹하는 픽스처."""
    with patch("src.pipeline.strategies.HwpParser") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestHwpParserStrategyInit:
    def test_creates_hwp_parser_instance(self):
        with patch("src.pipeline.strategies.HwpParser") as mock_cls:
            HwpParserStrategy()
            mock_cls.assert_called_once()


class TestHwpParserStrategyParse:
    def test_hwp_file_returns_section(self, tmp_path):
        """HWP 파일 파싱 시 is_raw_markdown 섹션 구조로 반환되는지 검증."""
        hwp_file = tmp_path / "규정.hwp"
        hwp_file.write_bytes(b"dummy hwp content")

        mock_parse_result = {"markdown": "# 제1조\n\n규정 내용입니다.", "extension": "hwp"}

        with patch("src.pipeline.strategies.HwpParser") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.parse.return_value = mock_parse_result
            mock_cls.return_value = mock_instance

            strategy = HwpParserStrategy()
            result = strategy.parse(hwp_file)

        assert len(result) == 1
        assert result[0]["is_raw_markdown"] is True
        assert "제1조" in result[0]["content"]
        assert result[0]["metadata"]["parser_type"] == "hwp"
        assert result[0]["metadata"]["doc_type"] == "hwp"
        assert result[0]["metadata"]["src_name"] == "규정.hwp"

    def test_hwpx_file_returns_section(self, tmp_path):
        """HWPX 파일 파싱 시 doc_type이 hwpx로 설정되는지 검증."""
        hwpx_file = tmp_path / "매뉴얼.hwpx"
        hwpx_file.write_bytes(b"dummy hwpx content")

        mock_parse_result = {
            "markdown": "# 매뉴얼\n\n| 항목 | 내용 |\n|------|------|\n| A | 1 |",
            "extension": "hwpx",
        }

        with patch("src.pipeline.strategies.HwpParser") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.parse.return_value = mock_parse_result
            mock_cls.return_value = mock_instance

            strategy = HwpParserStrategy()
            result = strategy.parse(hwpx_file)

        assert len(result) == 1
        assert result[0]["metadata"]["doc_type"] == "hwpx"

    def test_table_content_preserved(self, tmp_path):
        """표 구조가 Markdown pipe table 형식으로 보존되는지 검증."""
        hwp_file = tmp_path / "급여규정.hwp"
        hwp_file.write_bytes(b"dummy")

        table_markdown = "| 직급 | 기본급 |\n|------|--------|\n| 사원 | 300만원 |"
        mock_parse_result = {"markdown": table_markdown, "extension": "hwp"}

        with patch("src.pipeline.strategies.HwpParser") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.parse.return_value = mock_parse_result
            mock_cls.return_value = mock_instance

            strategy = HwpParserStrategy()
            result = strategy.parse(hwp_file)

        content = result[0]["content"]
        assert "| 직급 |" in content
        assert "| 사원 |" in content

    def test_unsupported_extension_returns_empty(self, tmp_path):
        """HWP/HWPX 외 확장자 입력 시 빈 리스트 반환 검증."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"dummy")

        with patch("src.pipeline.strategies.HwpParser"):
            strategy = HwpParserStrategy()
            result = strategy.parse(pdf_file)

        assert result == []

    def test_empty_markdown_returns_empty(self, tmp_path):
        """파싱 결과가 빈 문자열일 경우 빈 리스트 반환 검증."""
        hwp_file = tmp_path / "empty.hwp"
        hwp_file.write_bytes(b"dummy")

        mock_parse_result = {"markdown": "   ", "extension": "hwp"}

        with patch("src.pipeline.strategies.HwpParser") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.parse.return_value = mock_parse_result
            mock_cls.return_value = mock_instance

            strategy = HwpParserStrategy()
            result = strategy.parse(hwp_file)

        assert result == []

    def test_metadata_fields_present(self, tmp_path):
        """반환된 메타데이터에 필수 필드가 모두 존재하는지 검증."""
        hwp_file = tmp_path / "test.hwp"
        hwp_file.write_bytes(b"dummy")

        mock_parse_result = {"markdown": "내용이 있습니다.", "extension": "hwp"}

        with patch("src.pipeline.strategies.HwpParser") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.parse.return_value = mock_parse_result
            mock_cls.return_value = mock_instance

            strategy = HwpParserStrategy()
            result = strategy.parse(hwp_file)

        metadata = result[0]["metadata"]
        required_fields = ["source_id", "src_name", "relative_path", "parser_type", "doc_type", "pg_num"]
        for field in required_fields:
            assert field in metadata, f"필수 메타데이터 필드 누락: {field}"
