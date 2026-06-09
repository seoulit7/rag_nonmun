import sys; sys.stdout.reconfigure(encoding='utf-8')
import psycopg2
import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

# Consumer FK >= 10인 답변의 final_answer 조회
print("=" * 70)
print("Consumer FK >= 10 답변 내용 (idx 14,26,28,32)")
print("=" * 70)
cur.execute("""
    SELECT DISTINCT ON (query_index)
           query_index, disease, original_query, fk_grade, final_answer, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND expected_tier = 0
      AND user_level = 'Consumer'
      AND fk_grade >= 10
      AND query_index IN (14, 26, 28, 32)
    ORDER BY query_index, created_at DESC
""")
rows = cur.fetchall()
for r in rows:
    qi, dis, q, fk, ans, ts = r
    print(f"\n--- idx={qi} {dis} FK={fk:.2f} ---")
    print(f"Q: {q}")
    print(f"A: {ans[:800] if ans else '(없음)'}")

print()
print("=" * 70)
print("Fallback 케이스 내용 (idx 7,13,25,31 — Task1 대상)")
print("=" * 70)
cur.execute("""
    SELECT DISTINCT ON (query_index)
           query_index, disease, query_level_label, original_query,
           ragas_f, ragas_ar, ragas_cp, final_answer, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND expected_tier = 0
      AND is_fallback = TRUE
      AND query_level_label = 'P'
      AND query_index IN (7, 13, 25, 31)
    ORDER BY query_index, created_at DESC
""")
rows2 = cur.fetchall()
for r in rows2:
    qi, dis, ql, q, f, ar, cp, ans, ts = r
    print(f"\n--- idx={qi} {dis} [{ql}] F={f} AR={ar} CP={cp} ---")
    print(f"Q: {q}")
    print(f"A: {ans[:600] if ans else '(없음)'}")

conn.close()
