import json

import streamlit.components.v1 as components


def render_custom_copy_button(text_to_copy: str, key_suffix: str):
    """답변 복사 버튼을 렌더링한다.

    아이콘은 인라인 SVG로 렌더링한다(외부 폰트 CDN 의존 제거 — 오프라인/폐쇄망에서도 표시됨).
    고정 높이 iframe(`components.html`)로 렌더링해, st.html 컨테이너가 0 높이로 접혀 버튼이
    사라지던 문제를 원천 차단한다. 각 버튼은 격리된 iframe이라 메시지 간 스크립트 충돌도 없고,
    iframe은 자체 스크립트를 native로 실행하므로 복사 동작이 unsafe JS 허용 여부에 의존하지 않는다.
    """
    # 문자열 내 특수문자 및 줄바꿈으로 인한 스크립트 구문 오류 방지를 위한 이스케이프 처리
    safe_text = json.dumps(text_to_copy)

    html_code = f"""
    <style>
        body {{ margin: 0; }}
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
            const btn = document.getElementById('copybtn-{key_suffix}');
            const icCopy = document.getElementById('ic-copy-{key_suffix}');
            const icCheck = document.getElementById('ic-check-{key_suffix}');
            const text = {safe_text};
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
    # 고정 높이 iframe으로 렌더링(높이 0 접힘 방지). 버튼(아이콘 20px + 패딩)에 맞춘 최소 크기.
    components.html(html_code, height=36, width=44)
