import os
import re

import streamlit as st

import config.settings as settings
from models.state import GraphState
from ui.constants import TIER_CONFIGS
from ui.utils import score_badge, score_label


def _render_level(state: GraphState) -> None:
    level = state.get("user_level", "")
    label = (
        "👨‍⚕️ 의료 전문가 (Professional)"
        if level == "Professional"
        else "🙋 일반인 (Consumer)"
    )

    conf, intent, reasoning = 0.0, "기타", ""
    for line in reversed(state.get("log", [])):
        if "신뢰도=" in line:
            m_c = re.search(r"신뢰도=(\d+\.\d+)", line)
            m_i = re.search(r"의도=([^\s)]+)", line)
            conf = float(m_c.group(1)) if m_c else 0.0
            intent = m_i.group(1).rstrip(")") if m_i else "기타"
        if "[Level] 근거:" in line:
            reasoning = line.replace("[Level] 근거:", "").strip()

    st.markdown("**🧑‍⚕️ 사용자 수준 분류 완료**")
    cols = st.columns([2, 1, 1])
    cols[0].markdown(f"수준: {label}")
    cols[1].markdown(f"신뢰도: {score_badge(conf)}", unsafe_allow_html=True)
    cols[2].markdown(f"의도: `{intent}`")
    if reasoning:
        st.caption(f"📌 근거: {reasoning}")
    st.divider()


def _render_rewriter(state: GraphState) -> None:
    queries = state.get("queries", [])
    query = queries[-1] if queries else "-"
    loop = state.get("loop_count", 0)
    mode = f"재시도 {loop}회차 쿼리 개선" if loop > 0 else "최초 쿼리 최적화"

    reasoning = ""
    for line in reversed(state.get("log", [])):
        if "[Rewriter] 근거:" in line:
            reasoning = line.replace("[Rewriter] 근거:", "").strip()
            break

    st.markdown(f"**✏️ 쿼리 최적화 완료** ({mode})")
    st.code(query, language=None)
    if reasoning:
        st.caption(f"💡 {reasoning}")
    st.divider()


def _render_rag(state: GraphState) -> None:
    tier = state.get("search_tier", 0)
    cfg = TIER_CONFIGS.get(tier, {"name": "알 수 없음", "icon": "🔍", "desc": ""})
    sources = state.get("context_sources", [])
    chunks = state.get("context", [])

    st.markdown(f"**{cfg['icon']} 문서 검색 완료** — {cfg['name']}")
    st.caption(cfg["desc"])

    if sources:
        with st.expander(f"📂 검색된 소스 ({len(chunks)}개)", expanded=True):
            for i, (src, chunk) in enumerate(zip(sources, chunks), 1):
                if "#p" in src and os.path.exists(src.split("#p")[0]):
                    path_part, page_part = src.rsplit("#p", 1)
                    label = f"{os.path.basename(path_part)}  p.{int(page_part)+1}"
                else:
                    label = src
                st.markdown(f"**{i}.** `{label}`")
                st.caption(chunk[:200] + ("..." if len(chunk) > 200 else ""))
    st.divider()


def _render_critic(state: GraphState) -> None:
    f_score = state.get("critic_score", 0.0)
    ar = state.get("answer_relevance_score", 0.0)
    cp = state.get("context_precision_score", 0.0)
    flags = state.get("hallucination_flags", [])

    st.markdown("**🧪 RAGAS 품질 평가 완료**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Faithfulness (사실성)",    f"{f_score:.2f}", score_label(f_score),
              delta_color="normal" if f_score >= 0.8 else "inverse")
    c2.metric("Answer Relevance (관련성)", f"{ar:.2f}",      score_label(ar),
              delta_color="normal" if ar >= 0.8 else "inverse")
    c3.metric("Context Precision (정밀도)",f"{cp:.2f}",      score_label(cp),
              delta_color="normal" if cp >= 0.8 else "inverse")
    if flags:
        with st.expander(f"⚠️ 할루시네이션 감지 {len(flags)}건"):
            for flag in flags:
                st.warning(flag)
    st.divider()


def _render_tier_up(state: GraphState) -> None:
    tier = state.get("search_tier", 0)
    prev = TIER_CONFIGS.get(tier - 1, {"name": "?", "icon": "?"})
    curr = TIER_CONFIGS.get(tier,     {"name": "?", "icon": "?"})
    ar = state.get("answer_relevance_score", 0.0)

    st.warning(
        f"⬆️ **검색 소스 에스컬레이션**  \n"
        f"현재 AR={ar:.2f}  \n"
        f"참고: 즉시 에스컬은 AR이 {settings.CRITICAL_AR_THRESHOLD} 미만이거나, "
        f"F가 {settings.CRITICAL_F_THRESHOLD} 미만이면서 CP가 {settings.CRITICAL_CP_THRESHOLD} 미만일 때입니다.  \n"
        f"{prev['icon']} {prev['name']} → {curr['icon']} {curr['name']} 로 전환"
    )
    st.divider()


def _render_retry(state: GraphState) -> None:
    loop = state.get("loop_count", 0)
    f_score = state.get("critic_score", 0.0)
    ar = state.get("answer_relevance_score", 0.0)

    st.info(
        f"🔄 **재시도 {loop}/{settings.MAX_LOOPS}**  \n"
        f"Faithfulness {f_score:.2f} < {settings.FAITHFULNESS_THRESHOLD}  |  "
        f"Answer Relevance {ar:.2f}  \n"
        f"쿼리를 더 정밀하게 재최적화합니다."
    )
    st.divider()


def _render_output(state: GraphState) -> None:
    tier = state.get("search_tier", 0)
    name = TIER_CONFIGS.get(tier, {}).get("name", "?")
    st.success(
        f"📄 **최종 답변 생성 완료**  \n"
        f"최종 검색 소스: {name}  |  영문 → 한국어 번역 완료"
    )
    st.divider()


_RENDERERS = {
    "level":    _render_level,
    "rewriter": _render_rewriter,
    "rag":      _render_rag,
    "critic":   _render_critic,
    "tier_up":  _render_tier_up,
    "retry":    _render_retry,
    "output":   _render_output,
}


def on_step(step: str, state: GraphState) -> None:
    """LangGraph step_callback — 각 파이프라인 단계를 실시간 렌더링한다."""
    if step in _RENDERERS:
        _RENDERERS[step](state)
    elif step == "fallback":
        st.error("⚠️ **최대 재시도 초과** — 신뢰할 수 있는 근거를 찾지 못했습니다.")
        st.divider()
