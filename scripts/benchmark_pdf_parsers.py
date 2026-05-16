#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF 파서 벤치마킹 스크립트 (Option B: Docling 도입 평가)

목표:
1. Unstructured (현재) vs Docling (신규) 엔진 비교
2. Claude LLM Judge를 사용한 품질 평가
3. 텍스트 메트릭 (CER, 표 구조 복구율) 계산

실행: python scripts/benchmark_pdf_parsers.py
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키가 없으면 환경변수에서 읽기
if 'ANTHROPIC_API_KEY' not in os.environ:
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# 1. 의존성 확인
# ============================================================================

def check_dependencies():
    """필수 라이브러리 확인"""
    missing = []

    try:
        import anthropic
    except ImportError:
        missing.append("anthropic")

    if missing:
        print(f"⚠️  필수 라이브러리 누락: {', '.join(missing)}")
        print(f"설치: pip install {' '.join(missing)}")
        return False

    return True


# ============================================================================
# 2. Docling 파서
# ============================================================================

class DoclingPDFParser:
    """IBM Docling을 사용한 PDF 파서 (한글 최적화)"""

    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
        except ImportError as err:
            raise ImportError(
                "docling 라이브러리가 필요합니다.\n"
                "설치: pip install docling docling-core"
            ) from err

    def parse(self, file_path: str | Path) -> dict[str, Any]:
        """PDF 파싱 및 결과 반환"""
        start_time = time.time()

        result = self.converter.convert(str(file_path))
        doc = result.document

        # 전체 마크다운 추출
        full_text = doc.export_to_markdown()

        # 표 추출
        tables = self._extract_tables_from_markdown(full_text)

        parse_time = time.time() - start_time

        return {
            'text': full_text,
            'tables': tables,
            'metadata': {
                'total_pages': len(doc.pages) if hasattr(doc, 'pages') else 0,
                'parse_time': parse_time,
                'title': 'Docling parsed document',
                'engine': 'docling',
            }
        }

    @staticmethod
    def _extract_tables_from_markdown(markdown_text: str) -> list[dict[str, Any]]:
        """마크다운에서 표 추출"""
        tables = []
        lines = markdown_text.split('\n')
        current_table = []
        in_table = False

        for line in lines:
            if '|' in line:
                if not in_table:
                    in_table = True
                current_table.append(line)
            else:
                if in_table and current_table:
                    tables.append({
                        'markdown': '\n'.join(current_table),
                        'page': 'unknown',
                    })
                    current_table = []
                    in_table = False

        if current_table:
            tables.append({
                'markdown': '\n'.join(current_table),
                'page': 'unknown',
            })

        return tables


# ============================================================================
# 3. Unstructured 파서
# ============================================================================

class UnstructuredPDFParser:
    """
    Unstructured fast 모드 등가 파서: pdfminer.six 직접 사용

    [pdfminer 선택 근거]
    unstructured의 strategy="fast"는 내부적으로 pdfminer를 백엔드로 사용한다.
    그러나 unstructured 파이썬 패키지 자체가 unstructured_inference를
    import 시점에 자동 참조하여 ModuleNotFoundError가 발생한다.
    이에 pdfminer.high_level을 직접 호출하여 동등한 결과를 산출한다.
    (표 감지 불가는 unstructured fast와 동일한 제약)
    """

    def parse(self, file_path: str | Path) -> dict[str, Any]:
        start_time = time.time()

        from pdfminer.high_level import extract_text_to_fp, extract_pages
        from pdfminer.layout import LAParams, LTPage
        from io import StringIO

        # pdfminer 로 전체 텍스트 추출
        output = StringIO()
        with open(str(file_path), 'rb') as f:
            extract_text_to_fp(f, output, laparams=LAParams())
        full_text = output.getvalue()

        # 페이지 수 쪽산 (pdfminer 방식)
        page_count = sum(1 for _ in extract_pages(str(file_path)))

        parse_time = time.time() - start_time

        return {
            'text': full_text,
            'tables': [],  # pdfminer(=unstructured fast)는 표 구조 바로 추출 불가
            'metadata': {
                'total_pages': page_count,
                'parse_time': parse_time,
                'title': 'N/A',
                'engine': 'unstructured(pdfminer-fast)',
            }
        }


# ============================================================================
# 4. PyMuPDF 파서 (기준선)
# ============================================================================

class ManualPDFParser:
    """PyMuPDF (fitz) 기반 파서"""

    def parse(self, file_path: str | Path) -> dict[str, Any]:
        import fitz

        start_time = time.time()
        doc = fitz.open(str(file_path))
        text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        full_text = "\n\n".join(text_parts)
        parse_time = time.time() - start_time

        return {
            'text': full_text,
            'tables': [],
            'metadata': {
                'total_pages': len(doc),
                'parse_time': parse_time,
                'title': 'N/A',
                'engine': 'pymupdf',
            }
        }


# ============================================================================
# 5. 메트릭 계산
# ============================================================================

class TextMetrics:
    """텍스트 메트릭"""

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """Character Error Rate (CER) 계산"""
        from difflib import SequenceMatcher

        ref_norm = TextMetrics._normalize(reference)
        hyp_norm = TextMetrics._normalize(hypothesis)

        s = SequenceMatcher(None, ref_norm, hyp_norm)
        matching_chars = sum(block.size for block in s.get_matching_blocks())

        if len(ref_norm) == 0:
            return 0.0

        cer = 1.0 - (matching_chars / len(ref_norm))
        return round(cer * 100, 2)

    @staticmethod
    def calculate_table_preservation_rate(ref_tables: list, hyp_tables: list) -> float:
        """표 구조 보존율"""
        if len(ref_tables) == 0:
            return 100.0 if len(hyp_tables) == 0 else 0.0

        count_ratio = min(len(hyp_tables), len(ref_tables)) / len(ref_tables) * 100

        ref_chars = sum(t['markdown'].count('|') for t in ref_tables)
        hyp_chars = sum(t['markdown'].count('|') for t in hyp_tables)

        struct_ratio = (min(hyp_chars, ref_chars) / ref_chars * 100) if ref_chars > 0 else 100.0

        return round(count_ratio * 0.6 + struct_ratio * 0.4, 2)

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def count_korean_chars(text: str) -> int:
        return sum(1 for c in text if '\uac00' <= c <= '\ud7a3')


# ============================================================================
# 6. Claude LLM Judge
# ============================================================================

class LLMJudge:
    """Claude를 사용한 평가"""

    def __init__(self):
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()

            import anthropic
            # API 키 명시적으로 설정
            api_key = os.getenv('ANTHROPIC_API_KEY')
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logging.warning(f"Claude 초기화 실패: {e}")
            self.client = None

    def evaluate(self, text: str, tables: list, file_name: str) -> dict:
        """파싱 품질 평가"""

        if not self.client:
            return self._default()

        prompt = f"""당신은 PDF 파싱 엔진 평가 전문가입니다.

**문서:** {file_name}

다음 파싱 결과를 1-5점으로 평가해주세요:

**텍스트 샘플 (첫 1000자):**

{text[:1000]}
**표 개수:** {len(tables)}개

평가 항목 (각 1-5점):
1. 한글 무결성 (글자 깨짐 없음)
2. 구조 보존 (제목, 섹션, 계층)
3. 표 추출 품질
4. 전체 가독성
5. 종합 평가

**응답 형식 (JSON만):**
```json
{{"korean_integrity": <1-5>, "structure": <1-5>, "tables": <1-5>, "readability": <1-5>, "overall": <1-5>, "notes": "<의견>"}}
"""

        try:
            # 💡 .env 파일에서 EVAL_JUDGE_MODEL 값을 가져옵니다. (없으면 기본값 사용)
            import os
            judge_model = os.getenv("EVAL_JUDGE_MODEL", "claude-3-5-sonnet-latest")

            msg = self.client.messages.create(
                model=judge_model,  # 💡 하드코딩 대신 변수 사용
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response = msg.content[0].text
            # 정규식 수정: 백슬래시 제거
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)

            if match:
                return json.loads(match.group())
        except Exception as e:
            logging.warning(f"평가 실패: {e}")

        return self._default()

    @staticmethod
    def _default() -> dict:
        return {
            'korean_integrity': 0,
            'structure': 0,
            'tables': 0,
            'readability': 0,
            'overall': 0,
            'notes': '평가 미실행',
        }

class BenchmarkRunner:
    """벤치마크 실행"""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path('reports')
        self.output_dir.mkdir(exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run(self) -> dict:
        """벤치마크 실행"""

        self.logger.info(f"🚀 벤치마크 시작: {self.pdf_path.name}")

        results = {
            'file': self.pdf_path.name,
            'timestamp': time.time(),
            'parsers': {}
        }

        # 1. PyMuPDF
        self.logger.info("1/3: PyMuPDF 파싱...")
        pymupdf_result: Optional[dict] = None
        try:
            pymupdf_result = ManualPDFParser().parse(self.pdf_path)
            results['parsers']['pymupdf'] = self._evaluate(pymupdf_result)
        except Exception as e:
            self.logger.error(f"PyMuPDF 실패: {e}")

        # 2. Unstructured
        self.logger.info("2/3: Unstructured 파싱...")
        unstructured_result: Optional[dict] = None
        try:
            unstructured_result = UnstructuredPDFParser().parse(self.pdf_path)
            results['parsers']['unstructured'] = self._evaluate(unstructured_result)
        except Exception as e:
            self.logger.error(f"Unstructured 실패: {e}")

        # 3. Docling
        self.logger.info("3/3: Docling 파싱...")
        docling_result: Optional[dict] = None
        try:
            docling_result = DoclingPDFParser().parse(self.pdf_path)
            results['parsers']['docling'] = self._evaluate(docling_result)
        except Exception as e:
            self.logger.error(f"Docling 실패: {e}")

        # 비교 분석 (기준선: PyMuPDF, 비교 대상: Docling / Unstructured)
        # unstructured 실패 시에도 pymupdf vs docling 비교는 반드시 수행
        ref_result = pymupdf_result  # PyMuPDF를 기준(reference)으로 사용
        if ref_result and docling_result:
            results['comparison'] = {
                'reference_engine': 'pymupdf',
                'pymupdf_vs_docling': {
                    'cer': TextMetrics.calculate_cer(
                        ref_result['text'], docling_result['text']
                    ),
                    'table_preservation_docling': TextMetrics.calculate_table_preservation_rate(
                        docling_result['tables'], docling_result['tables']
                    ),
                    'table_count_pymupdf': len(ref_result['tables']),
                    'table_count_docling': len(docling_result['tables']),
                    'korean_chars_pymupdf': TextMetrics.count_korean_chars(ref_result['text']),
                    'korean_chars_docling': TextMetrics.count_korean_chars(docling_result['text']),
                },
            }
            # unstructured도 성공했으면 3자 비교 추가
            if unstructured_result:
                results['comparison']['pymupdf_vs_unstructured'] = {
                    'cer': TextMetrics.calculate_cer(
                        ref_result['text'], unstructured_result['text']
                    ),
                    'table_count_unstructured': len(unstructured_result['tables']),
                    'korean_chars_unstructured': TextMetrics.count_korean_chars(unstructured_result['text']),
                }

        self._save_report(results)
        return results

    def _evaluate(self, parse_result: dict) -> dict:
        """파서 결과 평가"""
        eval_result = {
            **parse_result['metadata'],
            'korean_chars': TextMetrics.count_korean_chars(parse_result['text']),
            'text_length': len(parse_result['text']),
            'table_count': len(parse_result['tables']),
        }

        judge = LLMJudge()
        claude_eval = judge.evaluate(
            parse_result['text'],
            parse_result['tables'],
            self.pdf_path.name
        )

        if claude_eval.get('overall'):
            eval_result['claude'] = claude_eval

        return eval_result

    def _save_report(self, results: dict):
        """리포트 저장"""
        timestamp = int(results['timestamp'])
        report_path = self.output_dir / f"benchmark_{timestamp}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"📁 리포트: {report_path}")
        self._print_summary(results)

    def _print_summary(self, results: dict):
        """결과 출력"""
        print("\n" + "=" * 100)
        print(f"📊 PDF 파서 벤치마크")
        print(f"파일: {results['file']}")
        print("=" * 100)

        for engine, data in results['parsers'].items():
            if data is None:
                print(f"\n❌ {engine.upper()}: 파싱 실패")
                continue

            print(f"\n📌 {engine.upper()}")
            print(f"  • 파싱 시간: {data.get('parse_time', 0):.2f}초")
            print(f"  • 페이지 수: {data.get('total_pages', 0)}")
            print(f"  • 텍스트 길이: {data.get('text_length', 0)}자")
            print(f"  • 한글 문자: {data.get('korean_chars', 0)}")
            print(f"  • 표 개수: {data.get('table_count', 0)}")

            if 'claude' in data:
                c = data['claude']
                if c.get('overall'):
                    print(f"  • Claude 평가:")
                    print(f"    - 한글 무결성: {c.get('korean_integrity', 0)}/5")
                    print(f"    - 구조 보존: {c.get('structure', 0)}/5")
                    print(f"    - 표 추출: {c.get('tables', 0)}/5")
                    print(f"    - 가독성: {c.get('readability', 0)}/5")
                    print(f"    - 종합: {c.get('overall', 0)}/5")
                    if c.get('notes'):
                        print(f"    - 의견: {c.get('notes')}")

        if 'comparison' in results:
            print(f"\n📈 Docling vs Unstructured 비교")
            print("=" * 100)
            comp = results['comparison']
            print(f"  • CER (문자 에러율): {comp.get('cer', 0):.2f}%")
            print(f"  • 표 구조 보존율: {comp.get('table_preservation', 0):.2f}%")

        print("\n" + "=" * 100 + "\n")

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)

    pdf_path = Path("data/raw/조선대학교_학칙.pdf")

    if not pdf_path.exists():
        print(f"❌ 파일 없음: {pdf_path}")
        sys.exit(1)

    runner = BenchmarkRunner(pdf_path)
    results = runner.run()

    print("✅ 벤치마크 완료!")
