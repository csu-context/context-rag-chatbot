import json

import streamlit as st


def render_custom_copy_button(text_to_copy: str, key_suffix: str):
    """답변 복사 버튼을 렌더링한다.

    인라인 HTML을 고정 높이 iframe으로 렌더한다. `components.v1.html`(2026-06-01 제거 예정
    deprecated)의 정식 후계자인 `st.iframe`을 사용한다 — HTML 문자열을 auto-detect해 srcdoc
    iframe으로 임베드한다(`<style>`로 시작시켜 URL/파일경로 오판 방지). iframe이라 컨테이너가
    0 높이로 접혀 버튼이 사라지지 않고, 스크립트도 native 실행되어 복사 동작이 unsafe JS 허용
    여부와 무관하다. 아이콘은 인라인 SVG로 외부 폰트 CDN 의존 없이 오프라인에서도 표시된다.
    """
    # 문자열 내 특수문자 및 줄바꿈으로 인한 스크립트 구문 오류 방지를 위한 이스케이프 처리
    safe_text = json.dumps(text_to_copy)

    # st.iframe의 입력 타입 auto-detect가 HTML로 분기하도록 반드시 '<'로 시작하게 한다.
    html_code = (
        "<style>"
        "  body { margin: 0; }"
        "  .copy-btn {"
        "    background: transparent; border: none; cursor: pointer; color: #555;"
        "    padding: 4px; border-radius: 4px; transition: background 0.2s;"
        "    display: inline-flex; align-items: center; justify-content: center;"
        "  }"
        "  .copy-btn:hover { background: #f0f0f0; }"
        "  .copy-btn svg { width: 20px; height: 20px; display: block; }"
        "</style>"
        f'<button class="copy-btn" id="copybtn-{key_suffix}" title="답변 복사하기">'
        f'  <span id="ic-copy-{key_suffix}" style="display:flex">'
        '    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"'
        '         stroke-linecap="round" stroke-linejoin="round">'
        '      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>'
        '      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>'
        "    </svg>"
        "  </span>"
        f'  <span id="ic-check-{key_suffix}" style="display:none; color:#4CAF50">'
        '    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"'
        '         stroke-linecap="round" stroke-linejoin="round">'
        '      <polyline points="20 6 9 17 4 12"></polyline>'
        "    </svg>"
        "  </span>"
        "</button>"
        "<script>"
        "  (function() {"
        f"    const btn = document.getElementById('copybtn-{key_suffix}');"
        f"    const icCopy = document.getElementById('ic-copy-{key_suffix}');"
        f"    const icCheck = document.getElementById('ic-check-{key_suffix}');"
        f"    const text = {safe_text};"
        "    function showCopied() {"
        "      icCopy.style.display = 'none'; icCheck.style.display = 'flex';"
        "      setTimeout(function() { icCopy.style.display = 'flex'; icCheck.style.display = 'none'; }, 2000);"
        "    }"
        "    function fallbackCopy() {"
        "      const ta = document.createElement('textarea');"
        "      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';"
        "      document.body.appendChild(ta); ta.select();"
        "      try { document.execCommand('copy'); showCopied(); }"
        "      catch (e) { console.error('Copy Failed', e); }"
        "      document.body.removeChild(ta);"
        "    }"
        "    btn.addEventListener('click', function() {"
        "      if (navigator.clipboard && navigator.clipboard.writeText) {"
        "        navigator.clipboard.writeText(text).then(showCopied).catch(fallbackCopy);"
        "      } else { fallbackCopy(); }"
        "    });"
        "  })();"
        "</script>"
    )

    # 고정 높이 iframe(접힘 방지). HTML 문자열이라 st.iframe이 srcdoc로 임베드한다.
    st.iframe(html_code, height=36, width=44)
