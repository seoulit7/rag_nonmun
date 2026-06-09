#!/usr/bin/env python
import sys
import json
import time

sys.stdout.reconfigure(encoding="utf-8")

from graph import run_medical_self_corrective_rag
from agents.critic import check_faithfulness
from agents.classifier import level_classifier

FAILED_INDICES = [2, 6, 12, 22, 30, 31]
LEVEL_MAP = {"P": "Professional", "C": "Consumer"}


def _load_questions():
    with open("main.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "QUESTIONS = [" in src and "assert len(QUESTIONS) == 40" in src:
            g = {}
            exec(src, g)
            return g["QUESTIONS"]
    raise RuntimeError("QUESTIONS 셀 없음")


def _is_fallback(state: dict) -> bool:
    if any("[Final] Fallback" in line for line in state.get("log", [])):
        return True
    return "신뢰할 수 있는 근거를 찾지 못했습니다" in (state.get("answer") or "")


def main():
    questions = _load_questions()
    passed = 0
    for idx in FAILED_INDICES:
        disease, level_label, exp_tier, question = questions[idx]
        qno = idx + 1
        expected = LEVEL_MAP[level_label]
        print(f"\n[{qno}] {disease} [{level_label}]")
        print(f"    Q: {question}")

        st = {"question": question, "user_level": "", "log": []}
        level_classifier(st)
        level_ok = st["user_level"] == expected
        print(f"    분류: {st['user_level']} ({'✓' if level_ok else '✗'})")

        t0 = time.perf_counter()
        state = run_medical_self_corrective_rag(
            question,
            forced_user_level=None,
            llm_provider="openai",
            ablation_condition="C",
            expected_tier=exp_tier,
            query_index=qno,
            disease=disease,
            query_level_label=level_label,
        )
        elapsed = time.perf_counter() - t0
        f = state.get("critic_score", 0.0)
        ar = state.get("answer_relevance_score", 0.0)
        cp = state.get("context_precision_score", 0.0)
        fb = _is_fallback(state)
        ok = check_faithfulness(state) and not fb and level_ok
        passed += int(ok)
        print(
            f"    {'✓ PASS' if ok else '✗ FAIL'} ({elapsed:.0f}s) "
            f"F={f:.3f} AR={ar:.3f} CP={cp:.3f}"
            f"{' [FALLBACK]' if fb else ''}"
        )

    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{len(FAILED_INDICES)} PASS")
    print(f"{'='*60}")
    return 0 if passed == len(FAILED_INDICES) else 1


if __name__ == "__main__":
    sys.exit(main())
