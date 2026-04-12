"""로그 조회 — 상세 화면."""
from __future__ import annotations

import streamlit as st

from ui.dashboard.log_query import TIER_LABEL, fetch_detail

_F_OK  = 0.8
_AR_OK = 0.7
_CP_OK = 0.8


def _score_badge(val: float, threshold: float, label: str) -> str:
    color = "#2ecc71" if val >= threshold else "#e74c3c"
    return (
        f"<span style='background:{color};color:#fff;"
        f"padding:2px 8px;border-radius:4px;font-size:0.85em'>"
        f"{label}={val:.3f}</span>"
    )


def render_detail(request_id: str) -> None:
    """단일 request_id 상세 화면을 렌더링한다."""

    # ── 뒤로가기 ─────────────────────────────────────────────────────────────
    if st.button("← 목록으로", type="secondary"):
        st.session_state["log_selected_id"] = None
        st.rerun()

    st.subheader("로그 상세")

    detail = fetch_detail(request_id)
    if not detail:
        st.error("데이터를 불러올 수 없습니다.")
        return

    meta         = detail["meta"]
    loops_df     = detail["loops"]
    queries      = detail["queries"]
    final_answer = detail["final_answer"]

    # ── 요청 정보 ─────────────────────────────────────────────────────────────
    st.markdown("#### 요청 정보")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("요청 ID",    meta["request_id"][:8] + "...")
    c2.metric("일시",       meta["created_at"])
    c3.metric("사용자 수준", meta["user_level"])
    c4.metric("LLM",        meta["llm_model"] or "—")

    st.markdown(
        f"**원본 질문** &nbsp; `{meta['original_query']}`",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── 루프 이력 테이블 ──────────────────────────────────────────────────────
    st.markdown("#### 루프 이력")

    show_cols = {
        "tier_label":       "Tier",
        "loop_count":       "Loop",
        "ragas_f":          "F",
        "ragas_ar":         "AR",
        "ragas_cp":         "CP",
        "is_escalated":     "에스컬",
        "is_fallback":      "Fallback",
        "retrieved_doc_count": "검색 청크",
        "execution_time_ms":   "평가(ms)",
        "created_at":       "일시",
    }
    loops_view = loops_df[list(show_cols.keys())].rename(columns=show_cols).copy()

    styled = (
        loops_view.style
        .applymap(lambda v: "color:#2ecc71" if isinstance(v, float) and v >= _F_OK  else
                            "color:#e74c3c" if isinstance(v, float) else "",
                  subset=["F"])
        .applymap(lambda v: "color:#2ecc71" if isinstance(v, float) and v >= _AR_OK else
                            "color:#e74c3c" if isinstance(v, float) else "",
                  subset=["AR"])
        .applymap(lambda v: "color:#2ecc71" if isinstance(v, float) and v >= _CP_OK else
                            "color:#e74c3c" if isinstance(v, float) else "",
                  subset=["CP"])
        .format({"F": "{:.3f}", "AR": "{:.3f}", "CP": "{:.3f}"}, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # 에스컬레이션 원인 자동 분석
    _render_escalation_summary(loops_df)

    st.markdown("---")

    # ── 최적화 쿼리 이력 ──────────────────────────────────────────────────────
    st.markdown("#### 최적화 쿼리 이력")
    if queries:
        for i, q in enumerate(queries, 1):
            st.markdown(f"**{i}회차** &nbsp; `{q}`", unsafe_allow_html=True)
    else:
        st.caption("쿼리 이력 없음")

    st.markdown("---")

    # ── 최종 답변 ─────────────────────────────────────────────────────────────
    st.markdown("#### 최종 답변")
    if final_answer:
        st.text_area(
            label="final_answer",
            value=final_answer,
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.caption("최종 답변이 아직 저장되지 않았습니다.")


def _render_escalation_summary(loops_df) -> None:
    """루프 이력에서 에스컬레이션 원인을 자동으로 분석해 표시한다."""
    msgs: list[str] = []
    for _, row in loops_df.iterrows():
        if not row.get("is_escalated"):
            continue
        tier  = row["tier_id"]
        ar    = row.get("ragas_ar", 0.0) or 0.0
        f_val = row.get("ragas_f",  0.0) or 0.0
        cp    = row.get("ragas_cp", 0.0) or 0.0
        loop  = row["loop_count"]

        if tier == 0 and ar < 0.3:
            msgs.append(f"Tier 0 · Loop {loop} — AR={ar:.3f} < 0.3 → 즉시 에스컬레이션")
        elif tier == 0:
            msgs.append(f"Tier 0 · Loop {loop} — 최대 루프 소진 (F={f_val:.3f}) → Tier 1 이동")
        elif tier == 1:
            msgs.append(f"Tier 1 → F={f_val:.3f} < 임계값 → Tier 2 이동")
        elif tier == 2:
            msgs.append(f"Tier 2 → F={f_val:.3f} 기준 미달 → Fallback")

    if msgs:
        with st.expander("에스컬레이션 분석", expanded=True):
            for m in msgs:
                st.markdown(f"- {m}")
