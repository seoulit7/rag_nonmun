#!/usr/bin/env python
import sys
import json
import time

sys.stdout.reconfigure(encoding="utf-8")

import config.settings as settings
from graph import run_medical_self_corrective_rag
from agents.critic import check_faithfulness
from agents.classifier import level_classifier

PATCHED_INDICES = [0, 2, 6, 8, 11, 12, 19, 22, 25, 30, 31]
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
    ans = state.get("answer") or ""
    return "신뢰할 수 있는 근거를 찾지 못했습니다" in ans


def _pass_scores(state: dict) -> bool:
    return check_faithfulness(state)


def main():
    questions = _load_questions()
    print("=" * 72)
    print("  Patched 11 — Classifier pre-check")
    print("=" * 72)
    cls_ok = 0
    for idx in PATCHED_INDICES:
        disease, level_label, exp_tier, question = questions[idx]
        expected = LEVEL_MAP[level_label]
        st = {"question": question, "user_level": "", "log": []}
        level_classifier(st)
        got = st["user_level"]
        ok = got == expected
        cls_ok += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{idx+1:2d}] {disease} [{level_label}] → {got} (기대 {expected})")
    print(f"  Classifier: {cls_ok}/{len(PATCHED_INDICES)}\n")

    print("=" * 72)
    print("  Patched 11 — Full pipeline (Condition C)")
    print("=" * 72)

    results = []
    for idx in PATCHED_INDICES:
        disease, level_label, exp_tier, question = questions[idx]
        qno = idx + 1
        print(f"\n[{qno:2d}] {disease} [{level_label}T{exp_tier}]")
        print(f"     Q: {question[:72]}...")

        t0 = time.perf_counter()
        try:
            state = run_medical_self_corrective_rag(
                question,
                forced_user_level=None,
                step_callback=None,
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
            scores_ok = _pass_scores(state)
            level_ok = state.get("user_level") == LEVEL_MAP[level_label]
            success = scores_ok and not fb and level_ok

            results.append(
                {
                    "idx": idx,
                    "disease": disease,
                    "level": level_label,
                    "success": success,
                    "f": f,
                    "ar": ar,
                    "cp": cp,
                    "fallback": fb,
                    "level_ok": level_ok,
                    "user_level": state.get("user_level"),
                    "elapsed": elapsed,
                    "error": None,
                }
            )
            marks = []
            if not fb:
                marks.append("no-fb")
            if scores_ok:
                marks.append("RAGAS✓")
            if level_ok:
                marks.append("level✓")
            print(
                f"     {'✓ PASS' if success else '✗ FAIL'} ({elapsed:.0f}s) "
                f"F={f:.3f} AR={ar:.3f} CP={cp:.3f} "
                f"level={state.get('user_level')} [{' '.join(marks) or '—'}]"
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append(
                {
                    "idx": idx,
                    "disease": disease,
                    "level": level_label,
                    "success": False,
                    "error": str(e),
                    "elapsed": elapsed,
                }
            )
            print(f"     ✗ ERROR ({elapsed:.0f}s): {str(e)[:100]}")

    passed = sum(1 for r in results if r.get("success"))
    fb_cnt = sum(1 for r in results if r.get("fallback"))
    ragas_cnt = sum(1 for r in results if r.get("success") or (r.get("f") is not None and _pass_scores({"critic_score": r["f"], "answer_relevance_score": r["ar"], "context_precision_score": r["cp"]})))
    errors = sum(1 for r in results if r.get("error"))

    print("\n" + "=" * 72)
    print(f"  Summary: PASS {passed}/{len(PATCHED_INDICES)} | Fallback {fb_cnt} | Errors {errors}")
    print(f"  Classifier pre-check: {cls_ok}/{len(PATCHED_INDICES)}")
    print("=" * 72)

    if passed < len(PATCHED_INDICES):
        print("\n  Failed items:")
        for r in results:
            if not r.get("success"):
                why = []
                if r.get("error"):
                    why.append(f"error={r['error'][:60]}")
                if r.get("fallback"):
                    why.append("fallback")
                if r.get("f") is not None and not _pass_scores(
                    {
                        "critic_score": r["f"],
                        "answer_relevance_score": r["ar"],
                        "context_precision_score": r["cp"],
                    }
                ):
                    why.append(f"RAGAS(F={r['f']:.2f},AR={r['ar']:.2f},CP={r['cp']:.2f})")
                if r.get("level_ok") is False:
                    why.append(f"level={r.get('user_level')}")
                print(f"    idx={r['idx']+1} {r['disease']} [{r['level']}]: {', '.join(why)}")

    return 0 if passed == len(PATCHED_INDICES) and cls_ok == len(PATCHED_INDICES) else 1


if __name__ == "__main__":
    sys.exit(main())
