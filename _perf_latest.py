import sys
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

FB13 = [3, 13, 25, 26, 31, 37, 38, 41, 44, 57, 61, 93, 101]

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

cur.execute(
    """
    SELECT MIN(created_at), MAX(created_at), COUNT(*)
    FROM public.rag_audit_log
    WHERE is_final = TRUE AND expected_tier = 0 AND ablation_condition = 'C'
    """
)
full_ts = cur.fetchone()

cur.execute(
    """
    SELECT query_index, disease, query_level_label,
           ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback,
           execution_time_ms, created_at
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, disease, query_level_label,
               ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback,
               execution_time_ms, created_at
        FROM public.rag_audit_log
        WHERE is_final = TRUE AND expected_tier = 0 AND ablation_condition = 'C'
        ORDER BY query_index, created_at DESC
    ) t
    ORDER BY query_index
    """
)
rows = cur.fetchall()

fb = [r for r in rows if r[7]]
fb13 = [r for r in rows if r[0] in FB13]
fb13_fb = [r for r in fb13 if r[7]]

pass_gate = sum(
    1
    for r in rows
    if not r[7] and (r[3] or 0) >= 0.8 and (r[4] or 0) >= 0.8 and (r[5] or 0) >= 0.8
)

conn.close()

fs = [r[3] for r in rows if r[3] is not None]
ars = [r[4] for r in rows if r[4] is not None]
cps = [r[5] for r in rows if r[5] is not None]
fks = [r[6] for r in rows if r[6] is not None]
ms = [r[8] for r in rows if r[8] is not None]

print("=== 조건 C Tier0 전체 102 (최신 is_final per query_index) ===")
print(f"로그 기간: {full_ts[0]} ~ {full_ts[1]}")
print(f"Fallback: {len(fb)}/102")
print(f"게이트 통과(F/AR/CP≥0.8, 비Fallback): {pass_gate}/102")
print(f"RAGAS 평균: F={mean(fs):.3f} AR={mean(ars):.3f} CP={mean(cps):.3f}")
print(f"FK 평균: {mean(fks):.2f}")
print(f"실행시간 평균: {mean(ms)/1000:.1f}s")
print()
print("=== 재작성 13건 cohort ===")
print(f"Fallback: {len(fb13_fb)}/13")
for r in fb13:
    qi, dis, ql, f, ar, cp, fk, isfb, ms, ts = r
    mark = "FB" if isfb else "OK"
    print(f"  [{mark}] idx={qi} {dis} {ql} F={f:.2f} AR={ar:.2f} CP={cp:.2f}")
print()
print("=== Fallback 15건 상세 ===")
for r in fb:
    qi, dis, ql, f, ar, cp, fk, isfb, ms, ts = r
    print(f"  idx={qi} {dis} {ql} F={f:.2f} AR={ar:.2f} CP={cp:.2f} FK={fk}")
