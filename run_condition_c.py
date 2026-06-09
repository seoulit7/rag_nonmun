#!/usr/bin/env python
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from graph import run_medical_self_corrective_rag
from stqs_questions import QUESTIONS, STQS_EXPECTED_TOTAL

assert len(QUESTIONS) == STQS_EXPECTED_TOTAL
print(f"Loaded {len(QUESTIONS)} questions from stqs_questions")

results = []
TOTAL = len(QUESTIONS)
print(f"\n{'═' * 75}")
print(f"  Condition C Re-run: No Multi-Tier ({len(QUESTIONS)} questions)")
print(f"{'═' * 75}\n")

for q_idx, (disease, level_label, exp_tier, question) in enumerate(QUESTIONS, 1):
    t0 = time.perf_counter()
    try:
        final_state = run_medical_self_corrective_rag(
            question,
            forced_user_level=None,
            step_callback=None,
            llm_provider="openai",
            ablation_condition="C",
            expected_tier=exp_tier,
            query_index=q_idx,
            disease=disease,
            query_level_label=level_label,
        )
        elapsed = time.perf_counter() - t0
        request_id = final_state.get("request_id", "")
        results.append((q_idx, disease, level_label, exp_tier, True, elapsed, request_id))

        status = "✓"
        fallback = "[FALLBACK]" if final_state.get("is_fallback") else ""
        print(
            f"  [{q_idx:3d}/{TOTAL}] {status} {disease:15s} "
            f"[{level_label}T{exp_tier}] {elapsed:5.1f}s {fallback}"
        )

    except Exception as e:
        elapsed = time.perf_counter() - t0
        results.append((q_idx, disease, level_label, exp_tier, False, elapsed, ""))
        print(
            f"  [{q_idx:3d}/{TOTAL}] ✗ {disease:15s} "
            f"[{level_label}T{exp_tier}] ERROR: {str(e)[:50]}"
        )

ok = sum(1 for r in results if r[4])
fail = TOTAL - ok
print(f"\n{'═' * 75}")
print(f"  Completed: {ok}/{TOTAL} successful")
print("  Results: Supabase rag_audit_log (condition='C', is_final=TRUE)")
print(f"{'═' * 75}\n")
