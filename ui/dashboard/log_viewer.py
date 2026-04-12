"""RAG 성능 대시보드 — 로그 조회 진입점.

log_selected_id 유무로 목록/상세 화면을 분기한다.
"""
import streamlit as st

from ui.dashboard.log_list   import render_list
from ui.dashboard.log_detail import render_detail


def render_log_viewer() -> None:
    """log_selected_id 세션 값에 따라 목록 또는 상세 화면을 표시한다."""
    if "log_selected_id" not in st.session_state:
        st.session_state["log_selected_id"] = None

    selected_id = st.session_state["log_selected_id"]

    if selected_id:
        render_detail(selected_id)
    else:
        render_list()
