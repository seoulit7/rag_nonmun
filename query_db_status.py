import sys; sys.stdout.reconfigure(encoding='utf-8')
import psycopg2
import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

print("=" * 70)
print("1) expected_tier=0 AND is_fallback=TRUE (최신 ablation 실행)")
print("=" * 70)
cur.execute("""
    SELECT query_index, disease, query_level_label, user_level,
           ablation_condition, original_query, ragas_f, ragas_ar, ragas_cp, fk_grade
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND expected_tier = 0
      AND is_fallback = TRUE
    ORDER BY ablation_condition, query_index
""")
rows = cur.fetchall()
print(f"  총 {len(rows)}건")
for r in rows:
    qi, dis, ql, ul, cond, q, f, ar, cp, fk = r
    print(f"  [{cond}] idx={qi} {dis} [{ql}/{ul}] F={f} AR={ar} CP={cp} FK={fk}")
    print(f"    Q: {q[:80]}")

print()
print("=" * 70)
print("2) Consumer fk_grade >= 9 (최신 ablation 실행, is_final=TRUE)")
print("=" * 70)
cur.execute("""
    SELECT query_index, disease, query_level_label, user_level,
           ablation_condition, original_query, fk_grade, ragas_ar
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND user_level = 'Consumer'
      AND fk_grade >= 9
    ORDER BY fk_grade DESC, query_index
""")
rows2 = cur.fetchall()
print(f"  총 {len(rows2)}건")
for r in rows2:
    qi, dis, ql, ul, cond, q, fk, ar = r
    print(f"  [{cond}] idx={qi} {dis} FK={fk} AR={ar}")
    print(f"    Q: {q[:80]}")

print()
print("=" * 70)
print("3) query_level_label vs user_level 불일치 (오분류, is_final=TRUE)")
print("=" * 70)
cur.execute("""
    SELECT query_index, disease, query_level_label, user_level,
           ablation_condition, original_query, ragas_f, ragas_ar
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND query_level_label IS NOT NULL
      AND query_level_label != ''
      AND (
        (query_level_label = 'P' AND user_level = 'Consumer')
        OR
        (query_level_label = 'C' AND user_level = 'Professional')
      )
    ORDER BY ablation_condition, query_index
""")
rows3 = cur.fetchall()
print(f"  총 {len(rows3)}건")
for r in rows3:
    qi, dis, ql, ul, cond, q, f, ar = r
    print(f"  [{cond}] idx={qi} {dis} 정답={ql} 분류={ul}")
    print(f"    Q: {q[:80]}")

print()
print("=" * 70)
print("4) 전체 Consumer FK 분포")
print("=" * 70)
cur.execute("""
    SELECT fk_grade, COUNT(*) as cnt
    FROM public.rag_audit_log
    WHERE is_final = TRUE AND user_level = 'Consumer' AND fk_grade IS NOT NULL
    GROUP BY fk_grade
    ORDER BY fk_grade
""")
rows4 = cur.fetchall()
for fk, cnt in rows4:
    bar = "█" * cnt
    print(f"  FK={fk:6.2f}: {bar} ({cnt}건)")

conn.close()
