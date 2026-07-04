from typing import Tuple

import streamlit as st

_REBUILD_STATUS = "rebuild_status"


def _init_rebuild_state() -> None:
    if _REBUILD_STATUS not in st.session_state:
        st.session_state[_REBUILD_STATUS] = "idle"


def render_sidebar() -> Tuple[str, str, str]:
    """Render the sidebar and return (user_persona, llm_backend, dashboard_menu).

    dashboard_menu: "Log Query" | "Performance Visualization" | "" (none selected)
    """
    _init_rebuild_state()

    st.sidebar.title("User Settings")
    user_persona = st.sidebar.selectbox(
        "User Persona", ["Auto Detect", "Medical Professional", "General User"]
    )

    if st.session_state.detected_level:
        label = (
            "Medical Professional"
            if st.session_state.detected_level == "Professional"
            else "General User"
        )
        st.sidebar.info(f"Auto-detected level: **{label}**")

    st.sidebar.markdown("---")
    llm_backend = "OpenAI"

    st.sidebar.markdown("---")
    _render_index_rebuilder()

    st.sidebar.markdown("---")
    dashboard_menu = _render_dashboard_menu()

    st.sidebar.markdown(
        "**Note**: This system is based on the MSD Manual and does not replace actual medical diagnosis or treatment."
    )

    return user_persona, llm_backend, dashboard_menu


def _render_dashboard_menu() -> str:
    """Render the RAG performance dashboard menu and return the selected sub-menu.

    Returns:
        "Log Query" | "Performance Visualization" | "" (none selected)
    """
    if "dashboard_menu" not in st.session_state:
        st.session_state["dashboard_menu"] = ""

    with st.sidebar.expander("📊 RAG Performance Dashboard", expanded=bool(st.session_state["dashboard_menu"])):
        if st.button("📋 Log Query", width="stretch",
                     type="primary" if st.session_state["dashboard_menu"] == "Log Query" else "secondary"):
            st.session_state["dashboard_menu"] = "Log Query"
            st.rerun()
        if st.button("📈 Performance Visualization", width="stretch",
                     type="primary" if st.session_state["dashboard_menu"] == "Performance Visualization" else "secondary"):
            st.session_state["dashboard_menu"] = "Performance Visualization"
            st.rerun()
        if st.session_state["dashboard_menu"]:
            st.markdown("---")
            if st.button("✕ Close", width="stretch"):
                st.session_state["dashboard_menu"] = ""
                st.rerun()

    return st.session_state["dashboard_menu"]


def _render_index_rebuilder() -> None:
    """Index rebuilder UI — synchronous execution on the main thread.

    Runs directly on the main thread without a background thread to avoid
    issues with module-level shared dicts being reset on each rerun in Streamlit Cloud.
    Results are stored in session_state.
    """
    st.sidebar.subheader("Index Rebuilder")
    st.sidebar.caption("Re-embeds all PDFs in the data folder and rebuilds the FAISS index from scratch.")

    status = st.session_state[_REBUILD_STATUS]

    if status == "idle":
        if st.sidebar.button("Rebuild Full Index", type="secondary", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "running"
            st.rerun()

    elif status == "running":
        progress_bar = st.sidebar.progress(0, text="Preparing...")
        status_text  = st.sidebar.empty()

        try:
            from tools.vector_search import rebuild_full_index

            def on_progress(pct: int, msg: str) -> None:
                progress_bar.progress(min(pct, 100) / 100, text=msg)
                status_text.caption(f"Rebuilding... {pct}%")

            n_pdfs, n_chunks = rebuild_full_index(on_progress=on_progress)
            st.session_state["rebuild_result"] = (n_pdfs, n_chunks)
            st.session_state[_REBUILD_STATUS]  = "done"

        except Exception as exc:
            st.session_state["rebuild_error"]  = str(exc)
            st.session_state[_REBUILD_STATUS]  = "error"

        st.rerun()

    elif status == "done":
        result = st.session_state.get("rebuild_result")
        if result:
            n_pdfs, n_chunks = result
            st.sidebar.success(f"Rebuild complete: {n_pdfs} PDFs · {n_chunks:,} chunks")
        if st.sidebar.button("OK", key="rebuild_done_btn", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "idle"
            st.rerun()

    elif status == "error":
        st.sidebar.error(f"Error: {st.session_state.get('rebuild_error', 'Unknown error')}")
        if st.sidebar.button("Close", key="rebuild_error_btn", width="stretch"):
            st.session_state[_REBUILD_STATUS] = "idle"
            st.rerun()
