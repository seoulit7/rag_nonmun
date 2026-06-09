import sys

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

FALLBACK_TRACK = sorted(
    {
        2, 3, 8, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 27, 32, 38, 39, 44, 45, 48,
        52, 53, 55, 56, 58, 59, 68, 71, 73, 76, 77, 80, 84, 90, 91, 97, 102, 103,
    }
)

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

cur.execute(
    """
    SELECT query_index, disease, query_level_label, original_query,
           ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback, execution_time_ms
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, disease, query_level_label, original_query,
               ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback, execution_time_ms
        FROM public.rag_audit_log
        WHERE is_final = TRUE AND expected_tier = 0 AND ablation_condition = 'C'
          AND query_index = ANY(%s)
        ORDER BY query_index, created_at DESC
    ) t
    WHERE is_fallback = TRUE
    ORDER BY query_index
    """,
    (FALLBACK_TRACK,),
)
fb = cur.fetchall()

cur.execute(
    """
    SELECT query_level_label,
           COUNT(*) FILTER (WHERE is_fallback) AS fb,
           COUNT(*) AS n,
           AVG(ragas_f) AS f, AVG(ragas_ar) AS ar, AVG(ragas_cp) AS cp
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, query_level_label, is_fallback, ragas_f, ragas_ar, ragas_cp
        FROM public.rag_audit_log
        WHERE is_final = TRUE AND expected_tier = 0 AND ablation_condition = 'C'
          AND query_index = ANY(%s)
        ORDER BY query_index, created_at DESC
    ) t
    GROUP BY query_level_label
    ORDER BY query_level_label
    """,
    (FALLBACK_TRACK,),
)
by_level = cur.fetchall()

conn.close()

print("=== Fallback 4건 상세 (38 추적 cohort) ===")
for r in fb:
    qi, dis, ql, q, f, ar, cp, fk, fb, ms = r
    print(f"idx={qi} {dis} [{ql}] F={f:.2f} AR={ar:.2f} CP={cp:.2f} FK={fk} {ms/1000:.1f}s")
    print(f"  Q: {q[:90]}...")

print()
print("=== P/C별 (38 추적 cohort) ===")
for r in by_level:
    ql, fb_n, n, f, ar, cp = r
    print(f"  [{ql}] {n}건, Fallback {fb_n}, F={f:.3f} AR={ar:.3f} CP={cp:.3f}")
