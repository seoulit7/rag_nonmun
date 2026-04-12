"""로그 조회 — 목록 화면."""
from __future__ import annotations

import math
from datetime import date, timedelta

import streamlit as st

from ui.dashboard.log_query import PAGE_SIZE, fetch_logs

# RAGAS 임계값 (색상 기준)
_F_OK  = 0.8
_AR_OK = 0.7
_CP_OK = 0.8


def _score_color(val: float, threshold: float) -> str:
    if val is None:
        return ""
    return "color: #2ecc71" if val >= threshold else "color: #e74c3c"


def render_list() -> None:
    """필터 패널 + 페이지네이션 + 테이블을 렌더링한다."""
    st.subheader("로그 조회")

    # ── 필터 패널 ────────────────────────────────────────────────────────────
    with st.expander("필터", expanded=True):
        col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])

        with col1:
            today = date.today()
            date_from = st.date_input("시작일", value=today - timedelta(days=7),
                                      key="lf_date_from")
            date_to   = st.date_input("종료일", value=today, key="lf_date_to")

        with col2:
            user_levels = st.multiselect(
                "사용자 수준",
                ["Professional", "Consumer"],
                default=[],
                key="lf_levels",
            )
            tiers = st.multiselect(
                "검색 Tier",
                options=[0, 1, 2],
                format_func=lambda x: {0: "0 · VectorDB", 1: "1 · LLM", 2: "2 · Web"}[x],
                default=[],
                key="lf_tiers",
            )

        with col3:
            escalated_opt = st.selectbox(
                "에스컬레이션", ["전체", "발생", "없음"], key="lf_escalated"
            )
            fallback_opt = st.selectbox(
                "Fallback", ["전체", "발생", "없음"], key="lf_fallback"
            )

        with col4:
            f_range = st.slider(
                "Faithfulness 범위",
                0.0, 1.0, (0.0, 1.0), step=0.05, key="lf_f_range",
            )
            keyword = st.text_input("키워드 (원본 질문)", key="lf_keyword")

        searched = st.button("조회", type="primary", width="content")

    # 조회 버튼 또는 초기 진입 시 실행
    if searched or "log_df" not in st.session_state:
        st.session_state["log_page"] = 1
        _run_query(date_from, date_to, user_levels, tiers,
                   escalated_opt, fallback_opt, f_range, keyword)

    df    = st.session_state.get("log_df")
    total = st.session_state.get("log_total", 0)

    if df is None or df.empty:
        st.info("조회 결과가 없습니다.")
        return

    # ── 건수 + 페이지 정보 ────────────────────────────────────────────────────
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page        = st.session_state.get("log_page", 1)

    st.caption(f"총 **{total:,}건** | 페이지 {page} / {total_pages}")

    # ── 테이블 ────────────────────────────────────────────────────────────────
    display_cols = {
        "created_at":       "일시",
        "request_id_short": "요청 ID",
        "user_level":       "수준",
        "original_query":   "원본 질문",
        "tier_label":       "Tier",
        "loop_count":       "Loop",
        "ragas_f":          "F",
        "ragas_ar":         "AR",
        "ragas_cp":         "CP",
        "is_escalated":     "에스컬",
        "is_fallback":      "Fallback",
        "execution_time_ms": "평가(ms)",
    }

    view = df[list(display_cols.keys())].rename(columns=display_cols).copy()
    # 원본 질문 truncate
    view["원본 질문"] = view["원본 질문"].str[:35] + "..."

    styled = (
        view.style
        .applymap(lambda v: _score_color(v, _F_OK),  subset=["F"])
        .applymap(lambda v: _score_color(v, _AR_OK), subset=["AR"])
        .applymap(lambda v: _score_color(v, _CP_OK), subset=["CP"])
        .format({"F": "{:.3f}", "AR": "{:.3f}", "CP": "{:.3f}"}, na_rep="—")
    )

    st.dataframe(styled, width="stretch", hide_index=True)

    # ── 행 선택 → 상세 이동 ──────────────────────────────────────────────────
    request_ids = df["request_id"].astype(str).tolist()
    id_options  = ["— 선택 —"] + [
        f"{df.iloc[i]['created_at']}  |  {df.iloc[i]['original_query'][:30]}..."
        for i in range(len(df))
    ]
    selected_idx = st.selectbox(
        "상세 보기 (행 선택)",
        range(len(id_options)),
        format_func=lambda i: id_options[i],
        key="log_select_idx",
    )
    if selected_idx > 0 and st.button("상세 보기", type="secondary"):
        st.session_state["log_selected_id"] = request_ids[selected_idx - 1]
        st.rerun()

    # ── 페이지네이션 ──────────────────────────────────────────────────────────
    st.markdown("---")
    pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
    with pcol1:
        if page > 1 and st.button("◀ 이전", width="stretch"):
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
        if page < total_pages and st.button("다음 ▶", width="stretch"):
            st.session_state["log_page"] = page + 1
            _run_query(date_from, date_to, user_levels, tiers,
                       escalated_opt, fallback_opt, f_range, keyword)
            st.rerun()


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _run_query(
    date_from, date_to, user_levels, tiers,
    escalated_opt, fallback_opt, f_range, keyword,
) -> None:
    escalated = None if escalated_opt == "전체" else (escalated_opt == "발생")
    fallback  = None if fallback_opt  == "전체" else (fallback_opt  == "발생")

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
