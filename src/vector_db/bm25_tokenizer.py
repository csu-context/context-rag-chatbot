import json
import logging

from kiwipiepy import Kiwi

from src.utils.paths import SYNONYMS_FILE

logger = logging.getLogger(__name__)

STOPWORDS: set[str] = {
    "대한",
    "대해",
    "위해",
    "통해",
    "경우",
    "또한",
    "모든",
    "의한",
    "따라",
    "기타",
    "사항",
    "있거나",
    "있으며",
    "의하여",
    "관하여",
    "다만",
}


class BM25Tokenizer:
    """한국어 형태소 분석(Kiwi) 및 동의어 처리를 담당하는 토크나이저"""

    def __init__(self):
        self.kiwi = Kiwi()
        self.synonyms = self._load_synonyms()
        self.stopwords = STOPWORDS

    def _load_synonyms(self) -> dict:
        if not SYNONYMS_FILE.exists():
            logger.warning(f"동의어 파일을 찾을 수 없습니다: {SYNONYMS_FILE} (빈 사전 사용)")
            return {}
        try:
            with open(SYNONYMS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"동의어 로드 중 오류 발생: {e}")
            return {}

    def _apply_synonyms(self, text: str) -> str:
        if not self.synonyms:
            return text
        for k, v in self.synonyms.items():
            # k가 v의 부분문자열이면 단순 substring 치환이 결과를 재오염시킨다.
            # 예: '조선대'→'조선대학교' 적용 시 '조선대학교'가 '조선대학교학교'로 파괴됨.
            # 이런 접두 중첩 약어는 형태소 경계 보장이 없으므로 치환에서 제외한다.
            if not k or k == v or k in v:
                continue
            text = text.replace(k, v)
        return text

    def tokenize(self, text: str) -> list[str]:
        """한국어 형태소 분석을 통해 의미 있는 토큰(명사, 용언 등)만 추출하고 불용어를 필터링함."""
        if not text:
            return []
        text = self._apply_synonyms(text)
        # N: 명사, V: 용언(동사/형용사), S: 외국어/숫자 추출 및 1글자 노이즈 제거
        tokens = [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith(("N", "V", "S")) and len(t.form) > 1]
        return [tok for tok in tokens if tok not in self.stopwords]
