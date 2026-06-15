import json

import streamlit as st


def render_custom_copy_button(text_to_copy: str, key_suffix: str):
    """클라이언트 브라우저에서 동작하는 자바스크립트 기반 커스텀 복사 버튼을 렌더링합니다.

    아이콘은 인라인 SVG로 렌더링한다(외부 폰트 CDN 의존 제거 — 오프라인/폐쇄망에서도 표시됨).
    st.html로 메인 DOM에 주입하고 메시지별 고유 id로 직접 바인딩해, 전역 함수 충돌(마지막
    메시지만 복사되던 문제)을 막는다.
    """
    # 문자열 내 특수문자 및 줄바꿈으로 인한 스크립트 구문 오류 방지를 위한 이스케이프 처리
    safe_text = json.dumps(text_to_copy)

    html_code = f"""
    <style>
        .copy-btn {{
            background: transparent;
            border: none;
            cursor: pointer;
            color: #555;
            padding: 4px;
            border-radius: 4px;
            transition: background 0.2s;
            display: flex;
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
    <script>
        (function() {{
            // st.html은 메인 DOM에 인라인 주입되므로(components.html의 iframe 격리 제거),
            // 메시지별 버튼에 고유 id로 직접 바인딩해 전역 함수 충돌(마지막 메시지만 복사)을 막는다.
            // 아이콘은 인라인 SVG로 렌더링한다(외부 폰트 CDN 의존 제거 — 오프라인/폐쇄망에서도 표시됨).
            const btn = document.getElementById('copybtn-{key_suffix}');
            if (!btn || btn.dataset.copyBound) return;  // 재실행 시 중복 바인딩 방지
            btn.dataset.copyBound = '1';
            const text = {safe_text};
            const icCopy = document.getElementById('ic-copy-{key_suffix}');
            const icCheck = document.getElementById('ic-check-{key_suffix}');
            btn.addEventListener('click', function() {{
                navigator.clipboard.writeText(text).then(function() {{
                    icCopy.style.display = 'none';
                    icCheck.style.display = 'flex';
                    setTimeout(function() {{
                        icCopy.style.display = 'flex';
                        icCheck.style.display = 'none';
                    }}, 2000);
                }}).catch(function(err) {{
                    console.error('Copy Failed', err);
                }});
            }});
        }})();
    </script>
    """
    st.html(html_code, unsafe_allow_javascript=True)
