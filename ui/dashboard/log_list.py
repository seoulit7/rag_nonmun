"""Log viewer — list screen."""
from __future__ import annotations

import math
from datetime import date, timedelta

import streamlit as st

from ui.dashboard.log_query import PAGE_SIZE, fetch_logs

_F_OK  = 0.8
_AR_OK = 0.8
_CP_OK = 0.8


def _score_color(val: float, threshold: float) -> str:
    if val is None:
        return ""
    return "color: #2ecc71" if val >= threshold else "color: #e74c3c"


def render_list() -> None:
    """Render the filter panel, pagination, and results table."""
    st.subheader("Log Query")

    with st.expander("Filters", expanded=True):
        col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])

        with col1:
            today = date.today()
            date_from = st.date_input("Start date", value=today - timedelta(days=7),
                                      key="lf_date_from")
            date_to   = st.date_input("End date", value=today, key="lf_date_to")

        with col2:
            user_levels = st.multiselect(
                "User Level",
                ["Professional", "Consumer"],
                default=[],
                key="lf_levels",
            )
            tiers = st.multiselect(
                "Search Tier",
                options=[0, 1, 2],
                format_func=lambda x: {0: "0 · VectorDB", 1: "1 · LLM", 2: "2 · Web"}[x],
                default=[],
                key="lf_tiers",
            )

        with col3:
            escalated_opt = st.selectbox(
                "Escalation", ["All", "Yes", "No"], key="lf_escalated"
            )
            fallback_opt = st.selectbox(
                "Fallback", ["All", "Yes", "No"], key="lf_fallback"
            )

        with col4:
            f_range = st.slider(
                "Faithfulness range",
                0.0, 1.0, (0.0, 1.0), step=0.05, key="lf_f_range",
            )
            keyword = st.text_input("Keyword (original query)", key="lf_keyword")

        searched = st.button("Search", type="primary", width="content")

    if searched or "log_df" not in st.session_state:
        st.session_state["log_page"] = 1
        _run_query(date_from, date_to, user_levels, tiers,
                   escalated_opt, fallback_opt, f_range, keyword)

    df    = st.session_state.get("log_df")
    total = st.session_state.get("log_total", 0)

    if df is None or df.empty:
        st.info("No results found.")
        return

    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page        = st.session_state.get("log_page", 1)

    st.caption(f"Total **{total:,}** records | Page {page} / {total_pages}")

    display_cols = {
        "created_at":        "Timestamp",
        "request_id_short":  "Request ID",
        "user_level":        "Level",
        "original_query":    "Original Query",
        "tier_label":        "Tier",
        "loop_number":       "Loop",
        "ragas_f":           "F",
        "ragas_ar":          "AR",
        "ragas_cp":          "CP",
        "is_escalated":      "Escalated",
        "is_fallback":       "Fallback",
        "execution_time_ms": "Time (ms)",
    }

    view = df[list(display_cols.keys())].rename(columns=display_cols).copy()
    view["Original Query"] = view["Original Query"].str[:35] + "..."

    styled = (
        view.style
        .applymap(lambda v: _score_color(v, _F_OK),  subset=["F"])
        .applymap(lambda v: _score_color(v, _AR_OK), subset=["AR"])
        .applymap(lambda v: _score_color(v, _CP_OK), subset=["CP"])
        .format({"F": "{:.3f}", "AR": "{:.3f}", "CP": "{:.3f}"}, na_rep="—")
    )

    st.dataframe(styled, width="stretch", hide_index=True)

    request_ids = df["request_id"].astype(str).tolist()
    id_options  = ["— Select —"] + [
        f"{df.iloc[i]['created_at']}  |  {df.iloc[i]['original_query'][:30]}..."
        for i in range(len(df))
    ]
    selected_idx = st.selectbox(
        "View Detail (select row)",
        range(len(id_options)),
        format_func=lambda i: id_options[i],
        key="log_select_idx",
    )
    if selected_idx > 0 and st.button("View Detail", type="secondary"):
        st.session_state["log_selected_id"] = request_ids[selected_idx - 1]
        st.rerun()

    st.markdown("---")
    pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
    with pcol1:
        if page > 1 and st.button("◀ Previous", width="stretch"):
            st.session_state["log_page"] = page - 1
            _run_query(date_from, date_to, user_levels, tiers,
                       escalated_opt, fallback_opt, f_range, keyword)
            st.rerun()
    with pcol2:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>{page} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with pcol3:
        if page < total_pages and st.button("Next ▶", width="stretch"):
            st.session_state["log_page"] = page + 1
            _run_query(date_from, date_to, user_levels, tiers,
                       escalated_opt, fallback_opt, f_range, keyword)
            st.rerun()


def _run_query(
    date_from, date_to, user_levels, tiers,
    escalated_opt, fallback_opt, f_range, keyword,
) -> None:
    escalated = None if escalated_opt == "All" else (escalated_opt == "Yes")
    fallback  = None if fallback_opt  == "All" else (fallback_opt  == "Yes")

    df, total = fetch_logs(
        date_from=date_from,
        date_to=date_to,
        user_levels=user_levels or None,
        tiers=tiers or None,
        escalated=escalated,
        fallback=fallback,
        ragas_f_min=f_range[0],
        ragas_f_max=f_range[1],
        keyword=keyword,
        page=st.session_state.get("log_page", 1),
    )
    st.session_state["log_df"]    = df
    st.session_state["log_total"] = total
