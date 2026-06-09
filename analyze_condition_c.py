#!/usr/bin/env python
"""Condition C 성능 분석 및 정리."""
import sys, psycopg2
sys.stdout.reconfigure(encoding='utf-8')

import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

# 1. 전체 요약
print("╔════════════════════════════════════════════════════════════════════════╗")
print("║                    Condition C 성능 분석 (Tier 0 중심)                  ║")
print("╚════════════════════════════════════════════════════════════════════════╝\n")

cur.execute("""
    SELECT
        COUNT(*) as total_runs,
        ROUND(AVG(ragas_f)::numeric, 3) as avg_f,
        ROUND(AVG(ragas_ar)::numeric, 3) as avg_ar,
        ROUND(AVG(ragas_cp)::numeric, 3) as avg_cp,
        SUM(CASE WHEN is_fallback THEN 1 ELSE 0 END) as fallback_count,
        MIN(created_at) as first_run,
        MAX(created_at) as last_run
    FROM public.rag_audit_log
    WHERE ablation_condition = 'C' AND expected_tier = 0 AND is_final = TRUE
""")
row = cur.fetchone()

if row:
    total, f, ar, cp, fallback, first, last = row
    print(f"📊 전체 통계 (expected_tier=0)")
    print(f"  총 실행: {total}회")
    print(f"  기간: {first.date()} ~ {last.date()}")
    print(f"\n📈 RAGAS 평가 지표 (목표: ≥0.8)")
    print(f"  F (Faithfulness):      {f:.3f} {'✓' if f >= 0.8 else '✗'}")
    print(f"  AR (Answer Relevance):  {ar:.3f} {'✓' if ar >= 0.8 else '✗'}")
    print(f"  CP (Context Precision): {cp:.3f} {'✓' if cp >= 0.8 else '✗'}")
    print(f"\n⚠️  Fallback Cases: {fallback}건 (목표: 0)")

# 2. Professional vs Consumer 비교
print(f"\n{'─'*76}")
cur.execute("""
    SELECT
        query_level_label,
        COUNT(*) as count,
        SUM(CASE WHEN is_fallback THEN 1 ELSE 0 END) as fallback,
        ROUND(AVG(ragas_f)::numeric, 3) as avg_f,
        ROUND(AVG(ragas_ar)::numeric, 3) as avg_ar,
        ROUND(AVG(ragas_cp)::numeric, 3) as avg_cp
    FROM public.rag_audit_log
    WHERE ablation_condition = 'C' AND expected_tier = 0 AND is_final = TRUE
    GROUP BY query_level_label
    ORDER BY query_level_label
""")
rows = cur.fetchall()

print(f"\n📋 수준별 성능 비교")
print(f"  {'수준':<10} {'건수':>5} {'Fallback':>9} {'F':>8} {'AR':>8} {'CP':>8}")
print(f"  {'-'*60}")
for level, count, fb, f, ar, cp in rows:
    level_name = "Professional" if level == "P" else "Consumer"
    print(f"  {level_name:<10} {count:>5} {fb or 0:>9} {f:.3f}  {ar:.3f}  {cp:.3f}")

# 3. Fallback 케이스 상세
print(f"\n{'─'*76}")
print(f"\n🚨 Fallback 케이스 상세 (expected_tier=0)")

cur.execute("""
    SELECT
        disease,
        query_level_label,
        expected_tier,
        COUNT(*) as count,
        ROUND(AVG(ragas_f)::numeric, 3) as avg_f,
        ROUND(AVG(ragas_ar)::numeric, 3) as avg_ar,
        ROUND(AVG(ragas_cp)::numeric, 3) as avg_cp,
        MAX(created_at) as latest
    FROM public.rag_audit_log
    WHERE ablation_condition = 'C' AND expected_tier = 0 AND is_final = TRUE
        AND is_fallback = TRUE
    GROUP BY disease, query_level_label, expected_tier
    ORDER BY disease
""")
fallback_rows = cur.fetchall()

if fallback_rows:
    print(f"  {'질환':<20} {'L':<3} {'F':>8} {'AR':>8} {'CP':>8}")
    print(f"  {'-'*60}")
    for disease, level, tier, count, f, ar, cp, latest in fallback_rows:
        print(f"  {disease:<20} {level:<3} {f:.3f}    {ar:.3f}    {cp:.3f}")
else:
    print(f"  ✓ Fallback 케이스 없음!")

# 4. 질환별 성능 (Tier 0만)
print(f"\n{'─'*76}")
print(f"\n🏥 질환별 성능 (expected_tier=0)")
cur.execute("""
    SELECT
        disease,
        COUNT(*) as runs,
        SUM(CASE WHEN is_fallback THEN 1 ELSE 0 END) as fallback,
        ROUND(AVG(ragas_f)::numeric, 3) as avg_f,
        ROUND(AVG(ragas_ar)::numeric, 3) as avg_ar,
        ROUND(AVG(ragas_cp)::numeric, 3) as avg_cp
    FROM public.rag_audit_log
    WHERE ablation_condition = 'C' AND expected_tier = 0 AND is_final = TRUE
    GROUP BY disease
    ORDER BY avg_f ASC, disease
""")
disease_rows = cur.fetchall()

print(f"  {'질환':<20} {'건수':>5} {'FB':>3} {'F':>8} {'AR':>8} {'CP':>8}")
print(f"  {'-'*60}")
for disease, runs, fb, f, ar, cp in disease_rows:
    fb_mark = "✓" if (fb or 0) > 0 else "-"
    print(f"  {disease:<20} {runs:>5} {fb_mark:>3} {f:.3f}    {ar:.3f}    {cp:.3f}")

conn.close()

print(f"\n{'═'*76}\n")
