from typing import Tuple

import streamlit as st

# session_state 키 상수
_REBUILD_STATUS = "rebuild_status"   # idle | running | done | error


def _init_rebuild_state() -> None:
    if _REBUILD_STATUS not in st.session_state:
        st.session_state[_REBUILD_STATUS] = "idle"


def render_sidebar() -> Tuple[str, str, str]:
    """사이드바를 렌더링하고 (user_persona, llm_backend, dashboard_menu)를 반환한다.

    dashboard_menu: "로그 조회" | "성능 시각화" | "" (선택 없음)
    """
    _init_rebuild_state()

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

    st.sidebar.markdown("---")
    _render_index_rebuilder()

    st.sidebar.markdown("---")
    dashboard_menu = _render_dashboard_menu()

    st.sidebar.markdown(
        "**주의**: 이 시스템은 MSD 매뉴얼 기반이며 실제 진단·치료를 대신하지 않습니다."
    )

    return user_persona, llm_backend, dashboard_menu


def _render_dashboard_menu() -> str:
    """RAG 성능 대시보드 메뉴를 렌더링하고 선택된 서브메뉴를 반환한다.

    Returns:
        "로그 조회" | "성능 시각화" | "" (선택 없음)
    """
    if "dashboard_menu" not in st.session_state:
        st.session_state["dashboard_menu"] = ""

    with st.sidebar.expander("📊 RAG 성능 대시보드", expanded=bool(st.session_state["dashboard_menu"])):
        if st.button("📋 로그 조회", width="stretch",
                     type="primary" if st.session_state["dashboard_menu"] == "로그 조회" else "secondary"):
            st.session_state["dashboard_menu"] = "로그 조회"
            st.rerun()
        if st.button("📈 성능 시각화", width="stretch",
                     type="primary" if st.session_state["dashboard_menu"] == "성능 시각화" else "secondary"):
            st.session_state["dashboard_menu"] = "성능 시각화"
            st.rerun()
        if st.session_state["dashboard_menu"]:
            st.markdown("---")
            if st.button("✕ 닫기", width="stretch"):
                st.session_state["dashboard_menu"] = ""
                st.rerun()

    return st.session_state["dashboard_menu"]


def _render_index_rebuilder() -> None:
    """인덱스 재빌더 UI — 메인 스레드 동기 실행 방식.

    Streamlit Cloud에서 모듈 레벨 공유 dict가 rerun마다 초기화되는 문제를 피하기 위해
    백그라운드 스레드 없이 메인 스레드에서 직접 실행한다.
    진행 결과는 session_state에 저장한다.
    """
    st.sidebar.subheader("인덱스 재빌더")
    st.sidebar.caption("data 폴더의 전체 PDF를 재임베딩하여 FAISS 인덱스를 새로 생성합니다.")

    status = st.session_state[_REBUILD_STATUS]

    # ── idle: 버튼 표시 ────────────────────────────────────────────────────
    if status == "idle":
        if st.sidebar.button("인덱스 전체 재빌드", type="secondary", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "running"
            st.rerun()

    # ── running: 동기 실행 ────────────────────────────────────────────────
    elif status == "running":
        progress_bar = st.sidebar.progress(0, text="준비 중...")
        status_text  = st.sidebar.empty()

        try:
            from tools.vector_search import rebuild_full_index

            def on_progress(pct: int, msg: str) -> None:
                progress_bar.progress(min(pct, 100) / 100, text=msg)
                status_text.caption(f"재빌드 중... {pct}%")

            n_pdfs, n_chunks = rebuild_full_index(on_progress=on_progress)
            st.session_state["rebuild_result"] = (n_pdfs, n_chunks)
            st.session_state[_REBUILD_STATUS]  = "done"

        except Exception as exc:
            st.session_state["rebuild_error"]  = str(exc)
            st.session_state[_REBUILD_STATUS]  = "error"

        st.rerun()

    # ── done: 완료 메시지 ──────────────────────────────────────────────────
    elif status == "done":
        result = st.session_state.get("rebuild_result")
        if result:
            n_pdfs, n_chunks = result
            st.sidebar.success(f"재빌드 완료: {n_pdfs}개 PDF · {n_chunks:,}개 청크")
        if st.sidebar.button("확인", key="rebuild_done_btn", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "idle"
            st.rerun()

    # ── error: 오류 메시지 ─────────────────────────────────────────────────
    elif status == "error":
        st.sidebar.error(f"오류: {st.session_state.get('rebuild_error', '알 수 없는 오류')}")
        if st.sidebar.button("닫기", key="rebuild_error_btn", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "idle"
            st.rerun()
