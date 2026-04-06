import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.state import GraphState
from utils.json_parser import parse_llm_json, fallback_optimizer_json
from core.llm_client import get_chat_llm, rewriter_model


_QUERY_OPTIMIZER_SYSTEM = """\
You are a medical RAG (Retrieval-Augmented Generation) query optimization expert.
Your task is to generate an optimal English search query for retrieving relevant passages
from the MSD Manual (a professional medical reference written entirely in English).

Rules:
- Output ONLY a JSON object, no other text.
- The query must be in English regardless of the input language.
- Use precise medical terminology appropriate for the user level.
- For Professional level: use clinical/pharmacological terms, include differential diagnoses or mechanisms if relevant.
- For Consumer level: use clear descriptive terms a patient would find in a medical reference.
- The query should be specific enough to retrieve targeted passages, not too broad.

JSON format:
{{
  "query": "<optimized English search query>",
  "reasoning": "<why this query will retrieve the best results (Korean, 1 sentence)>"
}}"""

_QUERY_REFINE_SYSTEM = """\
You are a medical RAG query refinement expert.
A previous search query failed to produce a high-quality answer.
Analyze the failure and generate an improved English search query for the MSD Manual.

Rules:
- Output ONLY a JSON object, no other text.
- The new query must be in English.
- Avoid repeating terms from failed queries that did not help.
- Try different angles: synonyms, related conditions, mechanisms, treatments, or symptoms.
- Be more specific or use alternative medical terminology.

JSON format:
{{
  "query": "<improved English search query>",
  "reasoning": "<why this new angle will work better (Korean, 1 sentence)>"
}}"""

_OPTIMIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QUERY_OPTIMIZER_SYSTEM),
    ("human", (
        "User question (may be in Korean): {question}\n"
        "User level: {user_level}\n"
        "Detected intent: {detected_intent}"
    )),
])

_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QUERY_REFINE_SYSTEM),
    ("human", (
        "Original question (may be in Korean): {question}\n"
        "User level: {user_level}\n"
        "Previously tried queries: {previous_queries}\n"
        "Last evaluation - Faithfulness: {faithfulness}, "
        "Answer Relevance: {answer_relevance}, "
        "Context Precision: {context_precision}\n"
        "Hallucination flags: {hallu_summary}\n\n"
        "Generate an improved query that addresses these failures."
    )),
])


def _optimize_query(question: str, user_level: str, detected_intent: str) -> dict:
    llm = get_chat_llm(model=rewriter_model(), temperature=0.2, max_tokens=1024)
    chain = _OPTIMIZE_PROMPT | llm.bind(response_format={"type": "json_object"}) | StrOutputParser()
    raw = chain.invoke({
        "question": question,
        "user_level": user_level,
        "detected_intent": detected_intent,
    })
    data = parse_llm_json(raw)
    if not (data.get("query") or "").strip():
        data.update(fallback_optimizer_json(raw))
    return data


def _refine_query(
    question: str,
    user_level: str,
    previous_queries: List[str],
    faithfulness: float,
    answer_relevance: float,
    context_precision: float,
    hallucination_flags: List[str],
) -> dict:
    hallu_summary = "; ".join(hallucination_flags[:3]) if hallucination_flags else "없음"
    llm = get_chat_llm(model=rewriter_model(), temperature=0.4, max_tokens=1024)
    chain = _REFINE_PROMPT | llm.bind(response_format={"type": "json_object"}) | StrOutputParser()
    raw = chain.invoke({
        "question": question,
        "user_level": user_level,
        "previous_queries": str(previous_queries),
        "faithfulness": f"{faithfulness:.2f}",
        "answer_relevance": f"{answer_relevance:.2f}",
        "context_precision": f"{context_precision:.2f}",
        "hallu_summary": hallu_summary,
    })
    data = parse_llm_json(raw)
    if not (data.get("query") or "").strip():
        data.update(fallback_optimizer_json(raw))
    return data


def adaptive_query_rewriter(state: GraphState) -> GraphState:
    """LLM 기반 쿼리 최적화 에이전트.

    - 최초 실행: 사용자 질문 + 수준 + 의도를 분석해 MSD Manual 검색에
      최적화된 영문 쿼리를 생성한다.
    - 재시도: 이전 쿼리의 실패 원인(Faithfulness, Answer Relevance, Context Precision,
      할루시네이션)을 LLM에 제공해 더 정확한 쿼리로 개선한다.
    """
    q = state["question"]
    level = state["user_level"]
    loop = state["loop_count"]

    if state.get("queries"):
        result = _refine_query(
            question=q,
            user_level=level,
            previous_queries=state["queries"],
            faithfulness=state.get("critic_score", 0.0),
            answer_relevance=state.get("answer_relevance_score", 0.0),
            context_precision=state.get("context_precision_score", 0.0),
            hallucination_flags=state.get("hallucination_flags", []),
        )
        mode = f"재시도 {loop}회차" if loop > 0 else f"Tier {state.get('search_tier', 0)} 에스컬레이션"
        state["log"].append(f"[Rewriter] {mode} - 쿼리 개선 모드")
    else:
        intent = "기타"
        for line in reversed(state.get("log", [])):
            if "의도=" in line:
                m = re.search(r"의도=([^\s)]+)", line)
                if m:
                    intent = m.group(1).rstrip(")")
                    break
        result = _optimize_query(question=q, user_level=level, detected_intent=intent)
        state["log"].append("[Rewriter] 최초 질의 최적화 모드")

    optimized_query = (result.get("query") or "").strip()
    reasoning = (result.get("reasoning") or "").strip()

    if not optimized_query:
        raise RuntimeError("LLM이 최적화된 쿼리를 생성하지 못했습니다.")

    state["queries"].append(optimized_query)
    state["log"].append(f"[Rewriter] 최적화 쿼리: '{optimized_query}'")
    if reasoning:
        state["log"].append(f"[Rewriter] 근거: {reasoning}")

    return state
