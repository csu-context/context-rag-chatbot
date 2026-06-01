"""KOREAN_NUMERIC_UNITS 외부 파일 병합·우선순위·정렬 검증 (리뷰 6번 후속).

도메인 종속 숫자 단위(학사 등)는 코드 기본값에서 제거되어 data/config/numeric_units.json
으로 분리되었다. 본 모듈은 (1) 파일 병합, (2) env 명시 우선, (3) 파일 부재 시 범용 폴백,
(4) 길이 내림차순 정렬을 고정한다.
"""

import json

from src.common.config import Settings, _order_units


def test_order_units_dedup_and_length_desc():
    # 중복 제거 + 길이 내림차순(긴 단위가 짧은 접두 단위보다 먼저 매칭되도록).
    ordered = _order_units(["학년", "학년도", "개", "학년", ""])
    assert ordered == ["학년도", "학년", "개"]
    assert ordered.index("학년도") < ordered.index("학년")


def test_numeric_units_merged_from_external_file(tmp_path, monkeypatch):
    # 파일에 도메인 단위를 주면 기본(범용) 단위에 병합되고 길이순 정렬된다.
    units_file = tmp_path / "numeric_units.json"
    units_file.write_text(json.dumps(["학기", "학년도"], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("NUMERIC_UNITS_PATH", str(units_file))

    s = Settings()
    assert "학기" in s.KOREAN_NUMERIC_UNITS
    assert "학년도" in s.KOREAN_NUMERIC_UNITS
    assert "개" in s.KOREAN_NUMERIC_UNITS  # 기본 범용 단위는 유지
    assert s.KOREAN_NUMERIC_UNITS.index("학년도") < s.KOREAN_NUMERIC_UNITS.index("개")


def test_env_explicit_overrides_and_skips_file(tmp_path, monkeypatch):
    # env로 KOREAN_NUMERIC_UNITS를 명시하면 파일은 무시되고 그 값이 전부다(명시 우선).
    units_file = tmp_path / "numeric_units.json"
    units_file.write_text(json.dumps(["학기"], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("NUMERIC_UNITS_PATH", str(units_file))
    monkeypatch.setenv("KOREAN_NUMERIC_UNITS", json.dumps(["개월", "개"], ensure_ascii=False))

    s = Settings()
    assert s.KOREAN_NUMERIC_UNITS == ["개월", "개"]  # 길이 내림차순, 파일 단위 미포함
    assert "학기" not in s.KOREAN_NUMERIC_UNITS


def test_missing_file_falls_back_to_generic_default(tmp_path, monkeypatch):
    # 파일이 없으면 코드 기본(범용 단위)만 사용한다. 학사 단위는 포함되지 않는다.
    monkeypatch.setenv("NUMERIC_UNITS_PATH", str(tmp_path / "does_not_exist.json"))

    s = Settings()
    assert "개" in s.KOREAN_NUMERIC_UNITS
    assert "학기" not in s.KOREAN_NUMERIC_UNITS
    assert "학년도" not in s.KOREAN_NUMERIC_UNITS


def test_malformed_file_falls_back_to_generic_default(tmp_path, monkeypatch):
    # 파일이 깨졌거나(JSON 오류) list가 아니면 경고 후 기본 단위만 사용한다(견고성).
    units_file = tmp_path / "numeric_units.json"
    units_file.write_text("{ not valid json", encoding="utf-8")
    monkeypatch.setenv("NUMERIC_UNITS_PATH", str(units_file))

    s = Settings()
    assert "개" in s.KOREAN_NUMERIC_UNITS
    assert "학기" not in s.KOREAN_NUMERIC_UNITS
