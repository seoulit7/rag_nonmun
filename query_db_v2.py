import sys; sys.stdout.reconfigure(encoding='utf-8')
import psycopg2
import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

print("=" * 70)
print("1) expected_tier=0 AND is_fallback=TRUE — 최신 실행 기준")
print("=" * 70)
cur.execute("""
    SELECT DISTINCT ON (query_index, ablation_condition)
           query_index, disease, query_level_label, user_level,
           ablation_condition, original_query, ragas_f, ragas_ar, ragas_cp, fk_grade,
           created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND expected_tier = 0
      AND is_fallback = TRUE
    ORDER BY query_index, ablation_condition, created_at DESC
""")
rows = cur.fetchall()
print(f"  총 {len(rows)}건")
for r in rows:
    qi, dis, ql, ul, cond, q, f, ar, cp, fk, ts = r
    print(f"  [{cond}] idx={qi} {dis} [{ql}/{ul}] F={f} AR={ar} CP={cp} FK={fk}")
    print(f"    Q: {q}")

print()
print("=" * 70)
print("2) expected_tier=0, Consumer, fk_grade >= 9 — 최신 실행 기준")
print("=" * 70)
cur.execute("""
    SELECT DISTINCT ON (query_index, ablation_condition)
           query_index, disease, query_level_label, user_level,
           ablation_condition, original_query, fk_grade, ragas_ar, ragas_f, ragas_cp,
           created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND expected_tier = 0
      AND user_level = 'Consumer'
      AND fk_grade >= 9
    ORDER BY query_index, ablation_condition, created_at DESC
""")
rows2 = cur.fetchall()
print(f"  총 {len(rows2)}건")
for r in rows2:
    qi, dis, ql, ul, cond, q, fk, ar, f, cp, ts = r
    print(f"  [{cond}] idx={qi} {dis} [{ql}/{ul}] FK={fk:.2f} F={f} AR={ar} CP={cp}")
    print(f"    Q: {q}")

print()
print("=" * 70)
print("3) query_level_label vs user_level 오분류 — 최신 실행 기준")
print("=" * 70)
cur.execute("""
    SELECT DISTINCT ON (query_index, ablation_condition)
           query_index, disease, query_level_label, user_level,
           ablation_condition, original_query, ragas_f, ragas_ar, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND query_level_label IS NOT NULL
      AND query_level_label != ''
      AND (
        (query_level_label = 'P' AND user_level = 'Consumer')
        OR
        (query_level_label = 'C' AND user_level = 'Professional')
      )
    ORDER BY query_index, ablation_condition, created_at DESC
""")
rows3 = cur.fetchall()
print(f"  총 {len(rows3)}건")
for r in rows3:
    qi, dis, ql, ul, cond, q, f, ar, ts = r
    print(f"  [{cond}] idx={qi} {dis} 정답={ql} 분류={ul} F={f} AR={ar}")
    print(f"    Q: {q}")

print()
print("=" * 70)
print("4) expected_tier=0 전체 결과 요약 (조건별 최신)")
print("=" * 70)
cur.execute("""
    SELECT DISTINCT ON (query_index, ablation_condition)
           query_index, disease, query_level_label, user_level,
           ablation_condition, ragas_f, ragas_ar, ragas_cp, fk_grade,
           is_fallback, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND expected_tier = 0
    ORDER BY query_index, ablation_condition, created_at DESC
""")
rows4 = cur.fetchall()
print(f"  총 {len(rows4)}건")
for r in rows4:
    qi, dis, ql, ul, cond, f, ar, cp, fk, fb, ts = r
    fb_str = " [FALLBACK]" if fb else ""
    print(f"  [{cond}] idx={qi} {dis} [{ql}/{ul}] F={f} AR={ar} CP={cp} FK={fk}{fb_str}")

conn.close()
