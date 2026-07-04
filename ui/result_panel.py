from typing import List

import streamlit as st


def render_result(result: str, search_tier: int, llm_provider: str) -> None:
    """Render the final answer and a tier-specific disclaimer."""
    st.subheader("Final Answer")
    st.write(result)

    if search_tier == 1:
        if llm_provider == "gemini":
            st.info(
                "This result is based on Gemini's pre-trained knowledge. "
                "For actual medical decisions, please consult a qualified healthcare professional."
            )
        else:
            st.info(
                "This result is based on GPT's pre-trained knowledge. "
                "For actual medical decisions, please consult a qualified healthcare professional."
            )
    elif search_tier == 2:
        st.info(
            "This result was generated from web search results. "
            "Please verify the source reliability and consult a qualified healthcare professional for clinical decisions."
        )
    else:
        st.info(
            "This result is based on the MSD Manual. "
            "For actual medical decisions, please consult a qualified healthcare professional."
        )


def render_log(logs: List[str]) -> None:
    """Render the detailed execution log expander."""
    with st.expander("🔎 Detailed Execution Log", expanded=False):
        for idx, line in enumerate(logs, 1):
            st.text(f"{idx:02d}. {line}")
