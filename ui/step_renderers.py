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
        "👨‍⚕️ Medical Professional"
        if level == "Professional"
        else "🙋 General Consumer"
    )

    conf, intent, reasoning = 0.0, "other", ""
    for line in reversed(state.get("log", [])):
        if "신뢰도=" in line:
            m_c = re.search(r"신뢰도=(\d+\.\d+)", line)
            m_i = re.search(r"의도=([^\s)]+)", line)
            conf = float(m_c.group(1)) if m_c else 0.0
            intent = m_i.group(1).rstrip(")") if m_i else "other"
        if "[Level] 근거:" in line:
            reasoning = line.replace("[Level] 근거:", "").strip()

    st.markdown("**🧑‍⚕️ User Level Classification Complete**")
    cols = st.columns([2, 1, 1])
    cols[0].markdown(f"Level: {label}")
    cols[1].markdown(f"Confidence: {score_badge(conf)}", unsafe_allow_html=True)
    cols[2].markdown(f"Intent: `{intent}`")
    if reasoning:
        st.caption(f"📌 Reasoning: {reasoning}")
    st.divider()


def _render_rewriter(state: GraphState) -> None:
    queries = state.get("queries", [])
    query = queries[-1] if queries else "-"
    loop = state.get("loop_count", 0)
    mode = f"Retry {loop} — query refinement" if loop > 0 else "Initial query optimization"

    reasoning = ""
    for line in reversed(state.get("log", [])):
        if "[Rewriter] Reasoning:" in line:
            reasoning = line.replace("[Rewriter] Reasoning:", "").strip()
            break
        if "[Rewriter] 근거:" in line:
            reasoning = line.replace("[Rewriter] 근거:", "").strip()
            break

    st.markdown(f"**✏️ Query Optimization Complete** ({mode})")
    st.code(query, language=None)
    if reasoning:
        st.caption(f"💡 {reasoning}")
    st.divider()


def _render_rag(state: GraphState) -> None:
    tier = state.get("search_tier", 0)
    cfg = TIER_CONFIGS.get(tier, {"name": "Unknown", "icon": "🔍", "desc": ""})
    sources = state.get("context_sources", [])
    chunks = state.get("context", [])

    st.markdown(f"**{cfg['icon']} Document Search Complete** — {cfg['name']}")
    st.caption(cfg["desc"])

    if sources:
        with st.expander(f"📂 Retrieved Sources ({len(chunks)})", expanded=True):
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
    st.markdown("**🧪 RAGAS Quality Evaluation Complete**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Faithfulness",     f"{f_score:.2f}", score_label(f_score),
              delta_color="normal" if f_score >= 0.8 else "inverse")
    c2.metric("Answer Relevance", f"{ar:.2f}",      score_label(ar),
              delta_color="normal" if ar >= 0.8 else "inverse")
    c3.metric("Context Precision", f"{cp:.2f}",     score_label(cp),
              delta_color="normal" if cp >= 0.8 else "inverse")
    st.divider()


def _render_tier_up(state: GraphState) -> None:
    tier = state.get("search_tier", 0)
    prev = TIER_CONFIGS.get(tier - 1, {"name": "?", "icon": "?"})
    curr = TIER_CONFIGS.get(tier,     {"name": "?", "icon": "?"})
    ar = state.get("answer_relevance_score", 0.0)

    st.warning(
        f"⬆️ **Search Source Escalation**  \n"
        f"Current AR={ar:.2f}  \n"
        f"Note: immediate escalation triggers when AR < {settings.CRITICAL_AR_THRESHOLD}, "
        f"or F < {settings.CRITICAL_F_THRESHOLD} AND CP < {settings.CRITICAL_CP_THRESHOLD}.  \n"
        f"{prev['icon']} {prev['name']} → {curr['icon']} {curr['name']}"
    )
    st.divider()


def _render_retry(state: GraphState) -> None:
    loop = state.get("loop_count", 0)
    f_score = state.get("critic_score", 0.0)
    ar = state.get("answer_relevance_score", 0.0)

    st.info(
        f"🔄 **Retry {loop}/{settings.MAX_LOOPS}**  \n"
        f"Faithfulness {f_score:.2f} < {settings.FAITHFULNESS_THRESHOLD}  |  "
        f"Answer Relevance {ar:.2f}  \n"
        f"Re-optimizing query with a more targeted approach."
    )
    st.divider()


def _render_output(state: GraphState) -> None:
    tier = state.get("search_tier", 0)
    name = TIER_CONFIGS.get(tier, {}).get("name", "?")
    st.success(
        f"📄 **Final answer generated**  \n"
        f"Source: {name}  |  English response"
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
    """LangGraph step_callback — renders each pipeline step in real time."""
    if step in _RENDERERS:
        _RENDERERS[step](state)
    elif step == "fallback":
        st.error("⚠️ **Max retries exceeded** — No reliable source found.")
        st.divider()
