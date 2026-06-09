#!/usr/bin/env python
import json
import sys
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8")

from graph import build_graph
from agents.critic import check_faithfulness
from infra.evaluator import get_pure_fk_grade, flesch_kincaid_grade_en

CONSUMER_T0 = [1, 3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 25, 27, 31, 33, 35, 37]
FK_MAX = 9.0


def _load_questions():
    with open("main.ipynb", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "QUESTIONS = [" in src and "assert len(QUESTIONS)" in src:
            g = {}
            exec(src, g)
            return g["QUESTIONS"]
    raise RuntimeError("QUESTIONS 없음")


def main():
    questions = _load_questions()
    graph = build_graph()
    results = []

    for idx in CONSUMER_T0:
        disease, level_label, exp_tier, question = questions[idx]
        qno = idx + 1
        print(f"\n[{qno}] {disease}")
        print(f"  Q: {question[:72]}")

        t0 = time.perf_counter()
        st = {}
        english = ""
        for event in graph.stream(
            {
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
            },
            stream_mode="updates",
        ):
            for node_name, updates in event.items():
                if isinstance(updates, dict):
                    st = {**st, **updates}
                if node_name in ("rag_engine", "critic") and st.get("answer"):
                    english = st["answer"]

        elapsed = time.perf_counter() - t0
        fk_masked = get_pure_fk_grade(english)
        fk_raw = flesch_kincaid_grade_en(english)
        f = st.get("critic_score", 0.0)
        ar = st.get("answer_relevance_score", 0.0)
        cp = st.get("context_precision_score", 0.0)
        fb = any("[Final] Fallback" in x for x in st.get("log", []))
        ragas_ok = check_faithfulness(st) and not fb
        fk_ok = fk_masked <= FK_MAX
        ok = fk_ok and ragas_ok

        results.append(
            {
                "idx": qno,
                "disease": disease,
                "fk": fk_masked,
                "fk_raw": fk_raw,
                "f": f,
                "ar": ar,
                "cp": cp,
                "ok": ok,
                "fk_ok": fk_ok,
                "ragas_ok": ragas_ok,
            }
        )
        print(
            f"  {'✓' if ok else '✗'} FK={fk_masked:.2f} (raw={fk_raw:.2f}) "
            f"F={f:.3f} AR={ar:.3f} CP={cp:.3f} ({elapsed:.0f}s)"
        )

    passed = sum(1 for r in results if r["ok"])
    fk_fail = [r for r in results if not r["fk_ok"]]
    ragas_fail = [r for r in results if not r["ragas_ok"]]

    print(f"\n{'='*60}")
    print(f"  PASS {passed}/{len(results)} (FK≤{FK_MAX} & RAGAS OK)")
    if fk_fail:
        print("  FK 초과:", ", ".join(f"idx={r['idx']} FK={r['fk']:.2f}" for r in fk_fail))
    if ragas_fail:
        print("  RAGAS/Fallback 실패:", ", ".join(f"idx={r['idx']}" for r in ragas_fail))
    print(f"{'='*60}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
