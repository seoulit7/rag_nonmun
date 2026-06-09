import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2

conn = psycopg2.connect(os.environ['SUPABASE_DB_URL'])
cur = conn.cursor()

print("=== 1. 조건별 × 수준별 fallback 현황 ===")
cur.execute("""
    SELECT ablation_condition, query_level_label,
           COUNT(*) FILTER (WHERE is_fallback=TRUE)  AS fallback_cnt,
           COUNT(*) FILTER (WHERE is_fallback=FALSE) AS ok_cnt,
           COUNT(*) AS total
    FROM rag_audit_log
    WHERE is_final = TRUE
    GROUP BY ablation_condition, query_level_label
    ORDER BY ablation_condition, query_level_label
""")
for r in cur.fetchall():
    rate = r[2]/r[4]*100 if r[4] else 0
    print(f"  [{r[0]}] {r[1]} | fallback={r[2]} / {r[4]} ({rate:.0f}%)")

print()
print("=== 2. 조건 C에서 Consumer fallback 상세 (RAGAS 점수 포함) ===")
cur.execute("""
    SELECT disease, ragas_f, ragas_ar, ragas_cp, self_correction_count, tier_path
    FROM rag_audit_log
    WHERE ablation_condition = 'C'
      AND query_level_label  = 'C'
      AND is_fallback = TRUE
      AND is_final    = TRUE
    ORDER BY ragas_ar
""")
rows = cur.fetchall()
print(f"  총 {len(rows)}건")
for r in rows:
    f  = r[1] or 0; ar = r[2] or 0; cp = r[3] or 0
    sc = r[4] or 0
    print(f"  {r[0]:16s} F={f:.2f} AR={ar:.2f} CP={cp:.2f} | 재시도={sc}회 | path={r[5]}")

print()
print("=== 3. 조건 C Consumer: 전체 RAGAS 평균 (fallback vs 성공) ===")
cur.execute("""
    SELECT is_fallback,
           ROUND(AVG(ragas_f)::numeric,3)  AS avg_f,
           ROUND(AVG(ragas_ar)::numeric,3) AS avg_ar,
           ROUND(AVG(ragas_cp)::numeric,3) AS avg_cp,
           COUNT(*) AS cnt
    FROM rag_audit_log
    WHERE ablation_condition = 'C'
      AND query_level_label  = 'C'
      AND is_final = TRUE
    GROUP BY is_fallback
    ORDER BY is_fallback
""")
for r in cur.fetchall():
    label = "fallback" if r[0] else "성공   "
    print(f"  {label} | F={r[1]} AR={r[2]} CP={r[3]} | n={r[4]}")

print()
print("=== 4. 조건 A(Full)와 C(No Multi-Tier) Consumer fallback 비교 ===")
cur.execute("""
    SELECT ablation_condition, is_fallback, COUNT(*) AS cnt
    FROM rag_audit_log
    WHERE query_level_label = 'C'
      AND is_final = TRUE
      AND ablation_condition IN ('A','C')
    GROUP BY ablation_condition, is_fallback
    ORDER BY ablation_condition, is_fallback
""")
for r in cur.fetchall():
    label = "fallback" if r[1] else "성공   "
    print(f"  [{r[0]}] {label}: {r[2]}건")

conn.close()
