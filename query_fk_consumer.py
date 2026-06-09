import sys

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

for threshold in (9, 8.5):
    cur.execute("""
        SELECT DISTINCT ON (query_index, ablation_condition)
               query_index, disease, ablation_condition, fk_grade,
               ragas_f, ragas_ar, ragas_cp, original_query
        FROM public.rag_audit_log
        WHERE is_final = TRUE
          AND expected_tier = 0
          AND user_level = 'Consumer'
          AND fk_grade >= %s
        ORDER BY query_index, ablation_condition, created_at DESC
    """, (threshold,))
    rows = cur.fetchall()
    print(f"\n=== fk_grade >= {threshold}: {len(rows)}건 ===")
    for r in rows:
        print(f"  [{r[2]}] idx={r[0]} {r[1]} FK={r[3]:.2f} F={r[4]:.3f} AR={r[5]:.3f} CP={r[6]:.3f}")
        print(f"    Q: {r[7][:90]}")

cur.execute("""
    SELECT query_index, disease, MIN(fk_grade), MAX(fk_grade), AVG(fk_grade)
    FROM public.rag_audit_log
    WHERE is_final = TRUE AND expected_tier = 0 AND user_level = 'Consumer'
      AND fk_grade IS NOT NULL
    GROUP BY query_index, disease
    ORDER BY MAX(fk_grade) DESC
""")
print("\n=== Consumer tier0 FK by query_index (max) ===")
for r in cur.fetchall():
    if r[3] and r[3] >= 8:
        print(f"  idx={r[0]} {r[1]} min={r[2]:.2f} max={r[3]:.2f} avg={r[4]:.2f}")

conn.close()
