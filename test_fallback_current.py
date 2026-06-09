#!/usr/bin/env python
import sys
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8")

from graph import build_graph
from agents.critic import check_faithfulness
from agents.classifier import level_classifier

INDICES = [1, 18]
LEVEL_MAP = {"P": "Professional", "C": "Consumer"}


def _load_questions():
    import json
    nb = json.load(open("main.ipynb", encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "QUESTIONS = [" in src and "assert len(QUESTIONS)" in src:
            g = {}
            exec(src, g)
            return g["QUESTIONS"]
    raise RuntimeError("QUESTIONS 없음")


def _is_fallback(state: dict) -> bool:
    if any("[Final] Fallback" in line for line in state.get("log", [])):
        return True
    return "신뢰할 수 있는 근거를 찾지 못했습니다" in (state.get("answer") or "")


def main():
    questions = _load_questions()
    graph = build_graph()
    ok_n = 0
    for idx in INDICES:
        disease, level_label, exp_tier, question = questions[idx]
        qno = idx + 1
        print(f"\n[{qno}] {disease} [{level_label}]")
        print(f"  Q: {question}")
        st_cls = {"question": question, "user_level": "", "log": []}
        level_classifier(st_cls)
        print(f"  분류: {st_cls['user_level']}")
        init = {
            "request_id": str(uuid.uuid4()),
            "question": question,
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
            "query_index": qno,
            "disease": disease,
            "query_level_label": level_label,
            "expected_tier": exp_tier,
        }
        st = {}
        t0 = time.perf_counter()
        for event in graph.stream(init, stream_mode="updates"):
            for node, upd in event.items():
                if isinstance(upd, dict):
                    st = {**st, **upd}
        elapsed = time.perf_counter() - t0
        f, ar, cp = st.get("critic_score", 0.0), st.get("answer_relevance_score", 0.0), st.get("context_precision_score", 0.0)
        fb = _is_fallback(st)
        ok = check_faithfulness(st) and not fb
        ok_n += int(ok)
        print(f"  {'✓ PASS' if ok else '✗ FAIL'} ({elapsed:.0f}s) F={f:.3f} AR={ar:.3f} CP={cp:.3f}{' [FB]' if fb else ''}")
    print(f"\nResult: {ok_n}/{len(INDICES)}")
    return 0 if ok_n == len(INDICES) else 1


if __name__ == "__main__":
    sys.exit(main())
