import sys

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()
cur.execute(
    """
    SELECT DISTINCT ON (query_index)
           query_index, disease, query_level_label, ablation_condition,
           original_query, ragas_f, ragas_ar, ragas_cp, created_at
    FROM public.rag_audit_log
    WHERE is_final = TRUE AND expected_tier = 0 AND is_fallback = TRUE
    ORDER BY query_index, created_at DESC
    """
)
rows = cur.fetchall()
conn.close()
for r in rows:
    print(repr(r))
