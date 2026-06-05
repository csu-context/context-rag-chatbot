import json
import streamlit.components.v1 as components


def render_custom_copy_button(text_to_copy: str, key_suffix: str):
    """클라이언트 브라우저 환경에서 동작하는 자바스크립트 기반의 커스텀 복사 버튼을 렌더링합니다.

    Material Symbols 사양의 아이콘을 활용하여 시각적 정합성을 확보합니다.
    """
    # 문자열 내 특수문자 및 줄바꿈으로 인한 스크립트 구문 오류 방지를 위한 이스케이프 처리
    safe_text = json.dumps(text_to_copy)

    # 웹 컴포넌트에 주입될 HTML, CSS, JavaScript 소스코드 정의
    html_code = f"""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
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
        .material-symbols-outlined {{
            font-size: 20px;
        }}
    </style>
    <button class="copy-btn" onclick="copyText()" title="답변 복사하기">
        <span class="material-symbols-outlined" id="icon-{key_suffix}">content_copy</span>
    </button>
    <script>
        function copyText() {{
            const text = {safe_text};
            // 비동기 Clipboard API를 통한 클라이언트 환경 복사 실행
            navigator.clipboard.writeText(text).then(function() {{
                const icon = document.getElementById('icon-{key_suffix}');
                icon.innerText = 'check';
                icon.style.color = '#4CAF50';
                // 2초 경과 후 원래의 content_copy 아이콘 상태로 원복 진행
                setTimeout(function() {{
                    icon.innerText = 'content_copy';
                    icon.style.color = '#555';
                }}, 2000);
            }}).catch(function(err) {{
                console.error('Copy Failed', err);
            }});
        }}
    </script>
    """
    # Streamlit 인라인 아이프레임 컴포넌트를 통한 마크업 독립 렌더링
    components.html(html_code, height=35, width=35)
