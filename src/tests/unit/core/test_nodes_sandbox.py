from langchain_core.documents import Document

from src.core.nodes import ContextBuilderNode


class TestContextBuilderSandbox:
    def test_escape_xml_replaces_entities(self):
        """& < > 가 HTML 엔티티로 치환된다."""
        assert ContextBuilderNode.escape_xml("<a> & </a>") == "&lt;a&gt; &amp; &lt;/a&gt;"

    def test_escape_xml_ampersand_first(self):
        """& 를 먼저 치환해 이중 이스케이프(&amp;lt;)를 방지한다."""
        assert ContextBuilderNode.escape_xml("&lt;") == "&amp;lt;"

    def test_format_docs_escapes_raw_xml(self):
        """문서 본문의 원시 XML 태그가 래퍼 밖으로 새어나오지 않는다(샌드박스 탈출 차단)."""
        docs = [Document(page_content="<x>&</x>", metadata={})]
        out = ContextBuilderNode.format_docs(docs)

        # 본문에서 온 원시 태그는 이스케이프되어 그대로 노출되지 않음
        assert "<x>" not in out
        assert "</x>" not in out
        assert "&lt;" in out and "&gt;" in out and "&amp;" in out
        # 우리가 생성한 래퍼 구조는 유지
        assert out.startswith('<document index="1">')
        assert out.rstrip().endswith("</document>")
