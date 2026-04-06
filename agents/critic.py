import re

import config.settings as settings
from models.state import GraphState
from infra.evaluator import compute_official_ragas_scores


_SUMMARY_PREFIX = re.compile(r"^\[(Consumer|Professional) Summary\]\s*", re.IGNORECASE)


def critic_agent(state: GraphState) -> GraphState:
    """RAGAS 공식 프레임워크 기반 3중 평가 에이전트."""
    # RAGAS 평가:
    # - 답변은 번역 전 영문 답변 사용 (prefix 제거)
    # - 질문은 영문 재작성 쿼리 사용 → 언어 불일치 방지 (영문 쿼리 ↔ 영문 답변)
    raw_answer = _SUMMARY_PREFIX.sub("", state["answer"]).strip()
    context_chunks = state["context"]
    eval_query = state["queries"][-1] if state["queries"] else state["question"]

    official = compute_official_ragas_scores(eval_query, raw_answer, context_chunks)

    state["critic_score"] = official.faithfulness
    state["answer_relevance_score"] = official.answer_relevance
    state["context_precision_score"] = official.context_precision
    state["hallucination_flags"] = official.hallu_flags

    state["log"].append(
        f"[Critic] RAGAS 공식 평가: "
        f"F={official.faithfulness:.3f}, AR={official.answer_relevance:.3f}, CP={official.context_precision:.3f}"
    )
    for flag in official.hallu_flags:
        state["log"].append(f"[Critic] {flag}")

    return state


def check_faithfulness(state: GraphState) -> bool:
    """F >= FAITHFULNESS_THRESHOLD AND AR >= AR_THRESHOLD AND CP >= CP_THRESHOLD 이면 Self-Correction Loop 종료.

    성공 기준:
      - F: 답변이 컨텍스트에 근거 (사실성)
      - AR: 답변이 질문에 충분히 관련 (관련성)
      - CP: 검색된 청크의 유효성 (정밀도)
    셋 다 충족해야 output으로 진행.
    """
    f_ok = state.get("critic_score", 0.0) >= settings.FAITHFULNESS_THRESHOLD
    ar_ok = state.get("answer_relevance_score", 0.0) >= settings.AR_THRESHOLD
    cp_ok = state.get("context_precision_score", 0.0) >= settings.CP_THRESHOLD
    return f_ok and ar_ok and cp_ok


def is_critically_low(state: GraphState) -> bool:
    """Tier 0 에서 RAGAS 지표가 현저히 낮아 query rewriting으로 개선 불가능한지 판단.

    즉시 에스컬레이션 조건 (OR):
    1. AR < CRITICAL_AR_THRESHOLD: VectorDB에 관련 내용이 아예 없음.
    2. F < CRITICAL_F_THRESHOLD AND CP < CRITICAL_CP_THRESHOLD: 검색 자체가 완전히 빗나감.
    """
    ar = state.get("answer_relevance_score", 0.0)
    f = state.get("critic_score", 0.0)
    cp = state.get("context_precision_score", 0.0)

    if ar < settings.CRITICAL_AR_THRESHOLD:
        return True
    if f < settings.CRITICAL_F_THRESHOLD and cp < settings.CRITICAL_CP_THRESHOLD:
        return True
    return False
