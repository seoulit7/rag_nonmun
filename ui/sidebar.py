from typing import Tuple

import streamlit as st


def render_sidebar() -> Tuple[str, str]:
    """사이드바를 렌더링하고 (user_persona, llm_backend)를 반환한다."""
    st.sidebar.title("사용자 설정")
    user_persona = st.sidebar.selectbox(
        "사용자 페르소나 선택", ["자동 분류", "의료 전문가", "일반인"]
    )

    if st.session_state.detected_level:
        label = (
            "의료 전문가"
            if st.session_state.detected_level == "Professional"
            else "일반인"
        )
        st.sidebar.info(f"자동 분류 결과: **{label}**")

    st.sidebar.markdown("---")
    llm_backend = st.sidebar.radio(
        "LLM 백엔드",
        ["OpenAI", "Gemini"],
        horizontal=True,
        help="Gemini 사용 시 GEMINI_API_KEY 필요. Google AI OpenAI 호환 API 사용.",
    )
    st.sidebar.markdown(
        "**주의**: 이 시스템은 MSD 매뉴얼 기반이며 실제 진단·치료를 대신하지 않습니다."
    )

    return user_persona, llm_backend
