from typing import Callable, Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.types import Command

import config.settings as settings
from models.state import GraphState
from tools.vector_search import initialize_vector_db
from agents.classifier import level_classifier
from agents.rewriter import adaptive_query_rewriter
from agents.rag_engine import rag_engine
from agents.critic import critic_agent, check_faithfulness, is_critically_low
from agents.output import output_agent
from core.llm_client import set_llm_provider, reset_llm_provider


# ── 노드 이름 → step_callback 이름 매핑 ─────────────────────────────────────
_NODE_TO_STEP = {
    "level_classifier": "level",
    "query_rewriter": "rewriter",
    "rag_engine": "rag",
    "critic": "critic",
    "output": "output",
    "fallback": "fallback",
}


# ── 노드 함수들 ──────────────────────────────────────────────────────────────

def _critic_node(
    state: GraphState,
) -> Command[Literal["query_rewriter", "output", "fallback"]]:
    """RAGAS 평가 후 Self-Corrective Loop 라우팅.

    - 기준 충족(F >= threshold) → output
    - Tier 0 재시도 가능 → query_rewriter (loop_count 증가)
    - Tier 0 즉시 에스컬레이션 또는 재시도 소진 → query_rewriter (search_tier=1)
    - Tier 1 기준 미달 → query_rewriter (search_tier=2)
    - Tier 2 기준 미달 → fallback
    """
    state = critic_agent(state)

    if check_faithfulness(state):
        return Command(update=dict(state), goto="output")

    tier = state["search_tier"]
    loop = state["loop_count"]
    ar = state.get("answer_relevance_score", 0.0)
    f = state.get("critic_score", 0.0)
    cp = state.get("context_precision_score", 0.0)

    new_state = {**state, "log": list(state["log"])}

    if tier == 0:
        if is_critically_low(state):
            new_state["search_tier"] = 1
            new_state["loop_count"] = 0
            new_state["log"].append(
                f"[Loop] RAGAS 지표 현저히 낮음 "
                f"(AR={ar:.2f}, F={f:.2f}, CP={cp:.2f}) → 즉시 Tier 1 에스컬레이션."
            )
        elif loop >= settings.MAX_LOOPS - 1:
            new_state["search_tier"] = 1
            new_state["loop_count"] = 0
            new_state["log"].append(
                f"[Loop] Tier 0 최대 재시도({settings.MAX_LOOPS}회) 소진 "
                f"(F={f:.2f}, AR={ar:.2f}) → Tier 1 에스컬레이션."
            )
        else:
            new_state["loop_count"] = loop + 1
            reasons = []
            if f < settings.FAITHFULNESS_THRESHOLD:
                reasons.append(f"F={f:.2f}<{settings.FAITHFULNESS_THRESHOLD}")
            if ar < settings.AR_THRESHOLD:
                reasons.append(f"AR={ar:.2f}<{settings.AR_THRESHOLD}")
            if cp < settings.CP_THRESHOLD:
                reasons.append(f"CP={cp:.2f}<{settings.CP_THRESHOLD}")
            new_state["log"].append(
                f"[Loop] Tier 0 재시도 {loop + 1}/{settings.MAX_LOOPS} — "
                f"{', '.join(reasons)} → query rewriting 재시도."
            )
        return Command(update=new_state, goto="query_rewriter")

    if tier == 1:
        new_state["search_tier"] = 2
        new_state["loop_count"] = 0
        new_state["log"].append(
            f"[Loop] Tier 1 기준 미달 (F={f:.2f} < {settings.FAITHFULNESS_THRESHOLD}) "
            "→ Tier 2 에스컬레이션."
        )
        return Command(update=new_state, goto="query_rewriter")

    # tier == 2 — 모든 Tier 소진
    return Command(update=new_state, goto="fallback")


def _fallback_node(state: GraphState) -> GraphState:
    """모든 Tier 소진 후 검색된 원문을 그대로 제시한다."""
    f = state.get("critic_score", 0.0)
    state["log"].append(
        f"[Final] 모든 Tier 소진 (최종 F={f:.2f}) — "
        "신뢰할 수 있는 근거를 찾지 못했습니다."
    )
    raw_ctx = (
        "\n\n---\n".join(state["context"]) if state["context"] else "(검색 결과 없음)"
    )
    state["answer"] = (
        "신뢰할 수 있는 근거를 찾지 못했습니다. "
        "아래는 검색된 원문 자료입니다. 직접 판단하시기 바랍니다.\n\n"
        f"[참고 원문]\n{raw_ctx}"
    )
    return output_agent(state)


# ── 그래프 빌드 ──────────────────────────────────────────────────────────────

def build_graph():
    """LangGraph StateGraph 기반 Self-Corrective RAG 그래프를 빌드·컴파일한다.

    노드 구성:
      level_classifier → query_rewriter → rag_engine → critic
                                  ↑________________________|  (Self-Corrective Loop)
      critic → output → END
      critic → fallback → END
    """
    graph = StateGraph(GraphState)

    # 노드 등록
    graph.add_node("level_classifier", level_classifier)
    graph.add_node("query_rewriter", adaptive_query_rewriter)
    graph.add_node("rag_engine", rag_engine)
    graph.add_node("critic", _critic_node)       # Command로 조건부 라우팅
    graph.add_node("output", output_agent)
    graph.add_node("fallback", _fallback_node)

    # 엣지 연결
    graph.set_entry_point("level_classifier")
    graph.add_edge("level_classifier", "query_rewriter")
    graph.add_edge("query_rewriter", "rag_engine")
    graph.add_edge("rag_engine", "critic")
    # critic → Command 반환: query_rewriter / output / fallback 으로 동적 라우팅
    graph.add_edge("output", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# 모듈 수준 싱글턴 (재컴파일 방지)
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── 메인 실행 함수 ────────────────────────────────────────────────────────────

def run_medical_self_corrective_rag(
    question: str,
    forced_user_level: Optional[str] = None,
    step_callback: Optional[Callable[[str, GraphState], None]] = None,
    llm_provider: str = "openai",
) -> GraphState:
    """Self-Corrective RAG 메인 실행 함수 (LangGraph 기반).

    실행 흐름:
      1. level_classifier — 사용자 수준 분류 (forced_user_level 시 스킵)
      2. query_rewriter   — MSD Manual 최적화 영문 쿼리 생성
      3. rag_engine       — ReAct 에이전트로 검색 + 답변 합성
      4. critic           — RAGAS 3중 평가 → Self-Corrective Loop 라우팅
      5. output           — 한국어 번역 + 출처·면책 조항 추가
      6. fallback         — 모든 Tier 소진 시 원문 제시
    """
    initialize_vector_db()

    prov = (llm_provider or "openai").strip().lower()
    if prov not in ("openai", "gemini"):
        prov = "openai"
    tok = set_llm_provider(prov)

    initial_state: GraphState = {
        "question": question,
        "user_level": forced_user_level or "",
        "queries": [],
        "context": [],
        "context_sources": [],
        "answer": "",
        "critic_score": 0.0,
        "answer_relevance_score": 0.0,
        "context_precision_score": 0.0,
        "hallucination_flags": [],
        "search_tier": 0,
        "llm_provider": prov,
        "loop_count": 0,
        "log": (
            [f"[Mode] 사용자 선택 레벨: {forced_user_level}."]
            if forced_user_level
            else []
        ),
    }

    try:
        graph = _get_graph()

        if step_callback is not None:
            # stream 모드: 노드 실행마다 step_callback 호출
            current_state = dict(initial_state)
            last_rewriter_tier = 0
            for event in graph.stream(initial_state, stream_mode="updates"):
                for node_name, updates in event.items():
                    pre_update_queries = list(current_state.get("queries") or [])
                    if isinstance(updates, dict):
                        current_state = {**current_state, **updates}
                    step = _NODE_TO_STEP.get(node_name, node_name)
                    # query_rewriter가 재호출(재시도/에스컬레이션)될 때 tier_up/retry 먼저 렌더링
                    if node_name == "query_rewriter" and pre_update_queries:
                        new_tier = current_state.get("search_tier", 0)
                        extra = "tier_up" if new_tier > last_rewriter_tier else "retry"
                        try:
                            step_callback(extra, current_state)
                        except Exception:
                            pass
                    try:
                        step_callback(step, current_state)
                    except Exception:
                        pass
                    if node_name == "query_rewriter":
                        last_rewriter_tier = current_state.get("search_tier", 0)
            return current_state

        return graph.invoke(initial_state)

    finally:
        reset_llm_provider(tok)
