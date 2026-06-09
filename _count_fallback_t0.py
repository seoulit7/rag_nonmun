import sys

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

cur.execute(
    """
    SELECT COUNT(*) FROM public.rag_audit_log
    WHERE expected_tier = 0 AND is_fallback = TRUE
    """
)
total_rows = cur.fetchone()[0]

cur.execute(
    """
    SELECT COUNT(*) FROM public.rag_audit_log
    WHERE expected_tier = 0 AND is_fallback = TRUE AND is_final = TRUE
    """
)
final_rows = cur.fetchone()[0]

cur.execute(
    """
    SELECT COUNT(*) FILTER (WHERE is_fallback) AS fb, COUNT(*) AS n
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, is_fallback
        FROM public.rag_audit_log
        WHERE expected_tier = 0 AND is_final = TRUE
        ORDER BY query_index, created_at DESC
    ) t
    """
)
fb_latest, n_latest = cur.fetchone()

cur.execute(
    """
    SELECT ablation_condition,
           COUNT(*) FILTER (WHERE is_fallback) AS fb,
           COUNT(*) AS n
    FROM (
        SELECT DISTINCT ON (query_index, ablation_condition)
               query_index, ablation_condition, is_fallback
        FROM public.rag_audit_log
        WHERE expected_tier = 0 AND is_final = TRUE
        ORDER BY query_index, ablation_condition, created_at DESC
    ) t
    GROUP BY ablation_condition
    ORDER BY ablation_condition
    """
)
by_cond = cur.fetchall()

cur.execute(
    """
    SELECT query_index, disease, query_level_label, ablation_condition,
           ragas_f, ragas_ar, ragas_cp
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, disease, query_level_label, ablation_condition,
               ragas_f, ragas_ar, ragas_cp, is_fallback, created_at
        FROM public.rag_audit_log
        WHERE expected_tier = 0 AND is_final = TRUE AND is_fallback = TRUE
        ORDER BY query_index, created_at DESC
    ) t
    ORDER BY query_index
    """
)
fb_list = cur.fetchall()

conn.close()

print("=== expected_tier=0 AND is_fallback=true ===")
print(f"전체 로그 행 수: {total_rows}")
print(f"is_final=TRUE 행 수: {final_rows}")
print(f"문항별 최신 is_final (distinct query_index): Fallback {fb_latest}/{n_latest}")
print()
print("=== ablation_condition별 (문항별 최신) ===")
for cond, fb, n in by_cond:
    print(f"  [{cond}] Fallback {fb}/{n}")
print()
print("=== Fallback 문항 목록 (최신 is_final) ===")
for r in fb_list:
    qi, dis, ql, cond, f, ar, cp = r
    print(f"  idx={qi} [{cond}] {dis} {ql} F={f:.2f} AR={ar:.2f} CP={cp:.2f}")
