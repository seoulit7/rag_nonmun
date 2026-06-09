import sys

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

MODIFIED_QUERY_INDICES = {
    1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 16, 20, 22, 23, 24, 26, 28, 31, 32, 34, 36, 38,
}

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

print("=== 1) 전체: is_fallback=TRUE AND fk_grade<=9 (최신 is_final per request) ===")
cur.execute("""
    SELECT DISTINCT ON (request_id)
           request_id, query_index, disease, user_level, ablation_condition,
           fk_grade, ragas_f, ragas_ar, ragas_cp, original_query, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND is_fallback = TRUE
      AND fk_grade IS NOT NULL
      AND fk_grade <= 9
    ORDER BY request_id, created_at DESC
""")
rows = cur.fetchall()
print(f"  총 {len(rows)}건")
for r in rows[:20]:
    print(f"  idx={r[1]} [{r[4]}] {r[2]} {r[3]} FK={r[5]:.2f} F={r[6]} AR={r[7]} CP={r[8]} @ {r[10]}")
    print(f"    Q: {(r[9] or '')[:70]}")

print("\n=== 2) 수정·테스트 대상 query_index만 (fk<=9 & fallback) ===")
cur.execute("""
    SELECT DISTINCT ON (query_index, ablation_condition)
           query_index, disease, user_level, ablation_condition,
           fk_grade, ragas_f, ragas_ar, ragas_cp, is_fallback, original_query, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND is_fallback = TRUE
      AND fk_grade IS NOT NULL
      AND fk_grade <= 9
      AND query_index = ANY(%s)
    ORDER BY query_index, ablation_condition, created_at DESC
""", (list(MODIFIED_QUERY_INDICES),))
rows2 = cur.fetchall()
print(f"  총 {len(rows2)}건")
for r in rows2:
    print(f"  idx={r[0]} [{r[3]}] {r[1]} {r[2]} FK={r[4]:.2f} F={r[5]} AR={r[6]} CP={r[7]} @ {r[10]}")
    print(f"    Q: {(r[9] or '')[:75]}")

print("\n=== 3) 수정 대상 중 최신 실행: fallback 여부 (조건 C) ===")
cur.execute("""
    SELECT DISTINCT ON (query_index)
           query_index, disease, user_level, is_fallback, fk_grade,
           ragas_f, ragas_ar, ragas_cp, original_query, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND ablation_condition = 'C'
      AND query_index = ANY(%s)
    ORDER BY query_index, created_at DESC
""", (list(MODIFIED_QUERY_INDICES),))
rows3 = cur.fetchall()
fb_low_fk = [r for r in rows3 if r[3] and r[4] is not None and r[4] <= 9]
fb_any = [r for r in rows3 if r[3]]
print(f"  최신 조건 C 실행: {len(rows3)}건, fallback={len(fb_any)}건")
print(f"  그중 fk_grade<=9 & fallback: {len(fb_low_fk)}건")
for r in fb_low_fk:
    print(f"    idx={r[0]} {r[1]} FK={r[4]:.2f} F={r[5]} AR={r[6]} CP={r[7]}")

conn.close()
