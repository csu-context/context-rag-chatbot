import json

import streamlit as st


def render_custom_copy_button(text_to_copy: str, key_suffix: str):
    """답변 복사 버튼을 렌더링한다.

    인라인 HTML 렌더링은 `st.html`로 한다(`components.v1.html`은 2026-06-01 제거 예정 deprecated,
    `st.iframe`은 URL 전용이라 인라인 HTML 불가). 아이콘은 인라인 SVG로 렌더링한다(외부 폰트 CDN
    의존 제거 — 오프라인/폐쇄망에서도 표시됨).

    st.html 컨테이너가 0 높이로 접혀 버튼이 사라지던 문제는, JS 보정 대신 CSS로 해당 컨테이너의
    min-height를 확정해(`:has()`로 이 버튼이 든 컨테이너만 한정) 자바스크립트 실행 여부와 무관하게
    항상 보이도록 한다. 복사는 Clipboard API + execCommand 폴백으로 처리한다.
    """
    # 문자열 내 특수문자 및 줄바꿈으로 인한 스크립트 구문 오류 방지를 위한 이스케이프 처리
    safe_text = json.dumps(text_to_copy)

    html_code = f"""
    <style>
        /* st.html 컨테이너가 0 높이로 접혀 버튼이 클리핑/소실되는 것을 CSS로 확정 방지(이 버튼 한정) */
        [data-testid="stHtml"]:has(#copybtn-{key_suffix}) {{
            min-height: 36px;
        }}
        .copy-wrapper {{
            min-height: 32px;
            display: flex;
            align-items: center;
        }}
        .copy-btn {{
            background: transparent;
            border: none;
            cursor: pointer;
            color: #555;
            padding: 4px;
            border-radius: 4px;
            transition: background 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }}
        .copy-btn:hover {{
            background: #f0f0f0;
        }}
        .copy-btn svg {{
            width: 20px;
            height: 20px;
            display: block;
        }}
    </style>
    <div class="copy-wrapper">
        <button class="copy-btn" id="copybtn-{key_suffix}" title="답변 복사하기">
            <span id="ic-copy-{key_suffix}" style="display:flex">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
            </span>
            <span id="ic-check-{key_suffix}" style="display:none; color:#4CAF50">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            </span>
        </button>
    </div>
    <script>
        (function() {{
            const btn = document.getElementById('copybtn-{key_suffix}');
            if (!btn || btn.dataset.copyBound) return;  // 재실행 시 중복 바인딩 방지
            btn.dataset.copyBound = '1';
            const text = {safe_text};
            const icCopy = document.getElementById('ic-copy-{key_suffix}');
            const icCheck = document.getElementById('ic-check-{key_suffix}');
            function showCopied() {{
                icCopy.style.display = 'none';
                icCheck.style.display = 'flex';
                setTimeout(function() {{
                    icCopy.style.display = 'flex';
                    icCheck.style.display = 'none';
                }}, 2000);
            }}
            function fallbackCopy() {{
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                try {{ document.execCommand('copy'); showCopied(); }}
                catch (e) {{ console.error('Copy Failed', e); }}
                document.body.removeChild(ta);
            }}
            btn.addEventListener('click', function() {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text).then(showCopied).catch(fallbackCopy);
                }} else {{
                    fallbackCopy();
                }}
            }});
        }})();
    </script>
    """
    st.html(html_code, unsafe_allow_javascript=True)
