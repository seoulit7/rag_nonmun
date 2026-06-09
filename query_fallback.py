import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2

conn = psycopg2.connect(os.environ['SUPABASE_DB_URL'])
cur = conn.cursor()

# Consumer 답변 중 FK >= 9인 개수
cur.execute("""
    SELECT COUNT(*)
    FROM rag_audit_log
    WHERE query_level_label = 'C'
      AND fk_grade >= 9
      AND is_final = TRUE
""")
total = cur.fetchone()[0]
print(f"Consumer(C) fk_grade >= 9, is_final=TRUE: {total}개")

# 상세 목록
cur.execute("""
    SELECT disease, fk_grade, ablation_condition, original_query
    FROM rag_audit_log
    WHERE query_level_label = 'C'
      AND fk_grade >= 9
      AND is_final = TRUE
    ORDER BY fk_grade DESC, ablation_condition
""")
rows = cur.fetchall()
for r in rows:
    fk = r[1] if r[1] is not None else 0.0
    q = r[3] if r[3] else ""
    print(f"  [{r[2]}] {r[0]} | FK={fk:.2f} | {q[:60]}")

conn.close()
