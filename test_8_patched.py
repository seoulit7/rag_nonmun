#!/usr/bin/env python
"""Test only the 8 patched questions to check for errors."""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')

import config.settings as settings
from graph import run_medical_self_corrective_rag

# Load QUESTIONS from main.ipynb
with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            if 'QUESTIONS = [' in src and 'assert len(QUESTIONS) == 40' in src:
                exec_globals = {}
                exec(src, exec_globals)
                QUESTIONS = exec_globals.get('QUESTIONS', [])
                break

assert len(QUESTIONS) == 40, f"Failed to load QUESTIONS"

# Only test 8 patched questions (0-indexed)
PATCHED_INDICES = [6, 8, 12, 22, 24, 25, 30, 31]

print(f"╔════════════════════════════════════════════════════════════════════════╗")
print(f"║  Testing 8 Patched Questions Only (Condition C)                        ║")
print(f"╚════════════════════════════════════════════════════════════════════════╝\n")

results = []
for idx in PATCHED_INDICES:
    q_idx_display = idx + 1  # 1-indexed for display
    disease, level_label, exp_tier, question = QUESTIONS[idx]

    print(f"[{q_idx_display}] {disease} [{level_label}T{exp_tier}]")
    print(f"    Q: {question[:70]}...")

    t0 = time.perf_counter()
    try:
        final_state = run_medical_self_corrective_rag(
            question,
            forced_user_level=None,
            step_callback=None,
            llm_provider="openai",
            ablation_condition="C",
            expected_tier=exp_tier,
            query_index=q_idx_display,
            disease=disease,
            query_level_label=level_label,
        )
        elapsed = time.perf_counter() - t0
        request_id = final_state.get("request_id", "")

        f_score = final_state.get("critic_score", 0)
        ar_score = final_state.get("answer_relevance_score", 0)
        cp_score = final_state.get("context_precision_score", 0)
        is_fallback = final_state.get("is_fallback", False)

        results.append((idx, disease, level_label, True, elapsed, f_score, ar_score, cp_score, is_fallback))

        fallback_mark = " [FALLBACK]" if is_fallback else ""
        print(f"    ✓ OK ({elapsed:.1f}s) F={f_score:.3f} AR={ar_score:.3f} CP={cp_score:.3f}{fallback_mark}\n")

    except Exception as e:
        elapsed = time.perf_counter() - t0
        results.append((idx, disease, level_label, False, elapsed, 0, 0, 0, False))
        print(f"    ✗ ERROR ({elapsed:.1f}s): {str(e)[:80]}\n")

# Summary
ok = sum(1 for r in results if r[3])
fail = len(PATCHED_INDICES) - ok
fallback = sum(1 for r in results if r[8])

print(f"╔════════════════════════════════════════════════════════════════════════╗")
print(f"║  Results: {ok}/{len(PATCHED_INDICES)} OK  |  {fail} Errors  |  {fallback} Fallback Cases")
print(f"╚════════════════════════════════════════════════════════════════════════╝\n")

if ok == len(PATCHED_INDICES):
    print("✓ All 8 patched questions executed successfully!\n")
    avg_f = sum(r[5] for r in results) / len(results)
    avg_ar = sum(r[6] for r in results) / len(results)
    avg_cp = sum(r[7] for r in results) / len(results)
    print(f"  Average metrics:")
    print(f"    F (Faithfulness):     {avg_f:.3f}")
    print(f"    AR (Answer Relevance): {avg_ar:.3f}")
    print(f"    CP (Context Precision):{avg_cp:.3f}")
    if fallback > 0:
        print(f"    Fallback cases: {fallback} (still present)")
else:
    print(f"✗ {fail} questions failed. Check errors above.\n")
