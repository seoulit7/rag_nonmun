import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from graph import build_graph
from agents.critic import check_faithfulness
from infra.evaluator import get_pure_fk_grade, flesch_kincaid_grade_en

q = "흡연이 심장 질환 위험을 높이는 이유는 무엇인가요?"
graph = build_graph()
init = {
    "request_id": "t",
    "question": q,
    "user_level": "",
    "queries": [],
    "context": [],
    "context_sources": [],
    "answer": "",
    "critic_score": 0.0,
    "answer_relevance_score": 0.0,
    "context_precision_score": 0.0,
    "hallucination_flags": [],
    "critic_feedback": "",
    "search_tier": 0,
    "loop_count": 0,
    "tier_path": "0",
    "self_correction_count": 0,
    "eval_count": 0,
    "best_answer": "",
    "best_q_total": 0.0,
    "llm_provider": "openai",
    "workflow_start_time": time.time(),
    "log": [],
    "ablation_condition": "C",
    "query_index": 6,
    "disease": "관상동맥질환",
    "query_level_label": "C",
    "expected_tier": 0,
}
st = {}
english = ""
for event in graph.stream(init, stream_mode="updates"):
    for node, upd in event.items():
        if isinstance(upd, dict):
            st = {**st, **upd}
        if node in ("rag_engine", "critic") and st.get("answer"):
            english = st["answer"]

fk = get_pure_fk_grade(english)
print(f"FK={fk:.2f} raw={flesch_kincaid_grade_en(english):.2f}")
print(f"F={st.get('critic_score')} AR={st.get('answer_relevance_score')} CP={st.get('context_precision_score')}")
print(f"RAGAS ok={check_faithfulness(st)}")
print(english[:500])
