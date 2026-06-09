#!/usr/bin/env python
"""Compare Condition C results: before patches vs after patches.
Shows improvement in fallback cases and RAGAS metrics.
"""
import sys, psycopg2
sys.stdout.reconfigure(encoding='utf-8')

import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

# Get all Condition C records, grouped by (disease, query_level_label, expected_tier)
# to compare old vs new
cur.execute("""
    SELECT
        disease,
        query_level_label,
        expected_tier,
        COUNT(*) as total_runs,
        COUNT(DISTINCT TO_DATE(created_at::text, 'YYYY-MM-DD')) as unique_days,
        AVG(CASE WHEN is_final THEN ragas_f ELSE NULL END) as avg_f,
        AVG(CASE WHEN is_final THEN ragas_ar ELSE NULL END) as avg_ar,
        AVG(CASE WHEN is_final THEN ragas_cp ELSE NULL END) as avg_cp,
        SUM(CASE WHEN is_final AND is_fallback THEN 1 ELSE 0 END) as fallback_count,
        SUM(CASE WHEN is_final THEN 1 ELSE 0 END) as final_count,
        MAX(created_at) as latest_run
    FROM public.rag_audit_log
    WHERE ablation_condition = 'C' AND is_final = TRUE
    GROUP BY disease, query_level_label, expected_tier
    ORDER BY disease, query_level_label
""")

rows = cur.fetchall()
conn.close()

if not rows:
    print("No Condition C results found.")
    sys.exit(0)

# Identify which are the patched 8 questions (indices 6,8,12,22,24,25,30,31)
# Map: idx → (disease, level)
PATCHED_8 = {
    6: ("뇌졸중", "P"),
    8: ("COPD", "P"),
    12: ("우울증", "P"),
    22: ("갑상선기능저하증", "P"),
    24: ("빈혈", "P"),
    25: ("빈혈", "C"),
    30: ("지역사회획득폐렴", "P"),
    31: ("지역사회획득폐렴", "C"),
}

patched_diseases_levels = set(PATCHED_8.values())

print("╔════════════════════════════════════════════════════════════════════════╗")
print("║  Condition C Results: Patched vs Non-Patched Questions                 ║")
print("╚════════════════════════════════════════════════════════════════════════╝\n")

patched_fallback_before = 0
patched_fallback_after = 0
patched_results = []
non_patched_results = []

for disease, level, tier, total, days, f, ar, cp, fallback, final, latest in rows:
    is_patched = (disease, level) in patched_diseases_levels

    record = {
        'disease': disease,
        'level': level,
        'tier': tier,
        'f': f or 0,
        'ar': ar or 0,
        'cp': cp or 0,
        'fallback': fallback or 0,
        'final': final or 0,
        'latest': latest,
    }

    if is_patched:
        patched_results.append(record)
    else:
        non_patched_results.append(record)

# Print patched questions
if patched_results:
    print("┌─ PATCHED (8建問題) ──────────────────────────────────────────────────┐")
    print(f"  {'Disease':<20} {'L':<3} {'T':<2} {'F':<7} {'AR':<7} {'CP':<7} {'Fallback':>8}")
    print("  " + "─" * 68)
    total_fallback = 0
    for r in sorted(patched_results, key=lambda x: (x['disease'], x['level'])):
        fb = "✓" if r['fallback'] > 0 else "-"
        if r['fallback'] > 0:
            total_fallback += r['fallback']
        print(f"  {r['disease']:<20} {r['level']:<3} {r['tier']:<2} "
              f"{r['f']:.3f}   {r['ar']:.3f}   {r['cp']:.3f}   "
              f"{r['fallback']:>2} {fb}")
    print(f"  {'Total Fallback':<20} {'':<3} {'':<2} {'':<7} {'':<7} {'':<7} {total_fallback:>3}")
    print("└" + "─" * 68 + "┘\n")

# Print non-patched for context
if non_patched_results:
    print("┌─ NON-PATCHED (32건) ─────────────────────────────────────────────────┐")
    print(f"  {'Disease':<20} {'L':<3} {'T':<2} {'F':<7} {'AR':<7} {'CP':<7} {'Fallback':>8}")
    print("  " + "─" * 68)
    total_fb = 0
    for r in sorted(non_patched_results, key=lambda x: (x['disease'], x['level'])):
        fb = "✓" if r['fallback'] > 0 else "-"
        if r['fallback'] > 0:
            total_fb += r['fallback']
        print(f"  {r['disease']:<20} {r['level']:<3} {r['tier']:<2} "
              f"{r['f']:.3f}   {r['ar']:.3f}   {r['cp']:.3f}   "
              f"{r['fallback']:>2} {fb}")
    print(f"  {'Total Fallback':<20} {'':<3} {'':<2} {'':<7} {'':<7} {'':<7} {total_fb:>3}")
    print("└" + "─" * 68 + "┘\n")

# Summary stats
if patched_results:
    p_fallback = sum(r['fallback'] for r in patched_results)
    p_avg_f = sum(r['f'] for r in patched_results) / len(patched_results)
    p_avg_ar = sum(r['ar'] for r in patched_results) / len(patched_results)
    p_avg_cp = sum(r['cp'] for r in patched_results) / len(patched_results)

    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║  PATCHED 8 Questions Summary                                           ║")
    print(f"║  Fallback Cases:  {p_fallback} (target: 0)                                        ║")
    print(f"║  Avg F (Faithfulness):     {p_avg_f:.3f} (target: ≥0.8)                      ║")
    print(f"║  Avg AR (Answer Relevance): {p_avg_ar:.3f} (target: ≥0.8)                      ║")
    print(f"║  Avg CP (Context Precision):{p_avg_cp:.3f} (target: ≥0.8)                      ║")
    print("╚════════════════════════════════════════════════════════════════════════╝\n")

print("✓ Run this after condition C completes to see improvement metrics.\n")
