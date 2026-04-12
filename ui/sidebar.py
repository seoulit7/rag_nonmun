import time
import threading
from typing import Tuple

import streamlit as st

# ── 스레드 간 공유 딕셔너리 (st.session_state 금지 구역) ─────────────────────
# 백그라운드 스레드는 이 딕셔너리에만 쓴다.
# 메인 스레드(Streamlit rerun)가 폴링 시 읽어서 session_state로 복사한다.
_shared: dict = {
    "pct":    0,
    "msg":    "",
    "result": None,   # (n_pdfs, n_chunks)
    "error":  "",
    "done":   False,
    "failed": False,
}

# session_state 키 상수 (메인 스레드 전용)
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
    """인덱스 재빌더 UI — 백그라운드 스레드 + 모듈 공유 딕셔너리 폴링 방식.

    핵심 원칙:
    - 백그라운드 스레드는 _shared 딕셔너리에만 쓴다 (st.session_state 접근 금지).
    - 메인 스레드(Streamlit rerun)가 2초마다 _shared를 읽어 UI를 갱신한다.
    """
    st.sidebar.subheader("인덱스 재빌더")
    st.sidebar.caption("data 폴더의 전체 PDF를 재임베딩하여 FAISS 인덱스를 새로 생성합니다.")

    status = st.session_state[_REBUILD_STATUS]

    # ── idle: 버튼 표시 ────────────────────────────────────────────────────
    if status == "idle":
        if st.sidebar.button(
            "인덱스 전체 재빌드", type="secondary", width="stretch"
        ):
            # 공유 딕셔너리 초기화 (메인 스레드에서만)
            _shared["pct"]    = 0
            _shared["msg"]    = "준비 중..."
            _shared["result"] = None
            _shared["error"]  = ""
            _shared["done"]   = False
            _shared["failed"] = False

            st.session_state[_REBUILD_STATUS] = "running"

            def _worker() -> None:
                """백그라운드 재빌드 — st.* API 일절 사용 금지."""
                try:
                    from tools.vector_search import rebuild_full_index

                    def on_progress(pct: int, msg: str) -> None:
                        # 일반 dict 쓰기만 수행 (thread-safe)
                        _shared["pct"] = pct
                        _shared["msg"] = msg

                    n_pdfs, n_chunks = rebuild_full_index(on_progress=on_progress)
                    _shared["result"] = (n_pdfs, n_chunks)
                    _shared["done"]   = True

                except Exception as exc:
                    _shared["error"]  = str(exc)
                    _shared["failed"] = True

            threading.Thread(target=_worker, daemon=True).start()
            st.rerun()

    # ── running: _shared 폴링 후 UI 갱신 ──────────────────────────────────
    elif status == "running":
        pct = _shared["pct"]
        msg = _shared["msg"] or "진행 중..."

        st.sidebar.progress(pct / 100, text=msg)
        st.sidebar.caption(f"백그라운드 재빌드 중... {pct}%")

        # 완료/실패 감지 → session_state 상태 전환 (메인 스레드에서만)
        if _shared["done"]:
            st.session_state[_REBUILD_STATUS] = "done"
        elif _shared["failed"]:
            st.session_state[_REBUILD_STATUS] = "error"

        time.sleep(2)   # 2초 후 rerun → 진행률 갱신
        st.rerun()

    # ── done: 완료 메시지 ──────────────────────────────────────────────────
    elif status == "done":
        result = _shared.get("result")
        if result:
            n_pdfs, n_chunks = result
            st.sidebar.success(f"재빌드 완료: {n_pdfs}개 PDF · {n_chunks:,}개 청크")
        if st.sidebar.button("확인", key="rebuild_done_btn", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "idle"
            st.rerun()

    # ── error: 오류 메시지 ─────────────────────────────────────────────────
    elif status == "error":
        st.sidebar.error(f"오류: {_shared.get('error', '알 수 없는 오류')}")
        if st.sidebar.button("닫기", key="rebuild_error_btn", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "idle"
            st.rerun()
