from typing import List

import streamlit as st


def render_result(result: str, search_tier: int, llm_provider: str) -> None:
    """최종 답변 본문과 tier별 면책 조항을 렌더링한다."""
    st.subheader("최종 결과")
    st.write(result)

    if search_tier == 1:
        if llm_provider == "gemini":
            st.info(
                "위 결과는 Gemini 학습 지식에 근거한 설명입니다. "
                "실제 의료 판단은 전문의와 상담하시기 바랍니다."
            )
        else:
            st.info(
                "위 결과는 GPT 사전 학습 데이터에 근거한 설명입니다. "
                "실제 의료 판단은 전문의와 상담하시기 바랍니다."
            )
    elif search_tier == 2:
        st.info(
            "위 결과는 웹 검색을 바탕으로 생성되었습니다. "
            "출처를 확인하시고, 실제 의료 판단은 전문의와 상담하시기 바랍니다."
        )
    else:
        st.info(
            "위 결과는 MSD 매뉴얼을 기반으로 생성된 답변입니다. "
            "실제 의료 판단은 전문의와 상담하시기 바랍니다."
        )


def render_log(logs: List[str]) -> None:
    """상세 실행 로그 expander를 렌더링한다."""
    with st.expander("🔎 상세 실행 로그", expanded=False):
        for idx, line in enumerate(logs, 1):
            st.text(f"{idx:02d}. {line}")
