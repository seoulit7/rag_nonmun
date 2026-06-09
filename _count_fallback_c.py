import sys

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

FALLBACK_TRACK = {
    2, 3, 8, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 27, 32, 38, 39, 44, 45, 48,
    52, 53, 55, 56, 58, 59, 68, 71, 73, 76, 77, 80, 84, 90, 91, 97, 102, 103,
}

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()
cur.execute(
    """
    SELECT query_index, is_fallback, created_at
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, is_fallback, created_at
        FROM public.rag_audit_log
        WHERE is_final = TRUE
          AND expected_tier = 0
          AND ablation_condition = 'C'
          AND query_index = ANY(%s)
        ORDER BY query_index, created_at DESC
    ) t
    ORDER BY query_index
    """,
    (list(FALLBACK_TRACK),),
)
rows = cur.fetchall()
conn.close()

logged = {r[0] for r in rows}
fb = [r[0] for r in rows if r[1]]
missing = sorted(FALLBACK_TRACK - logged)
print(f"tracked_indices: {len(FALLBACK_TRACK)}")
print(f"logged_latest: {len(logged)}")
print(f"fallback_latest: {len(fb)} {fb}")
print(f"not_logged_yet: {len(missing)} {missing[:20]}{'...' if len(missing)>20 else ''}")
