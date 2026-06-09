import sys
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import config.settings as settings

FALLBACK_TRACK = sorted(
    {
        2, 3, 8, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 27, 32, 38, 39, 44, 45, 48,
        52, 53, 55, 56, 58, 59, 68, 71, 73, 76, 77, 80, 84, 90, 91, 97, 102, 103,
    }
)


def agg(rows):
    if not rows:
        return {}
    fs = [r[0] for r in rows if r[0] is not None]
    ars = [r[1] for r in rows if r[1] is not None]
    cps = [r[2] for r in rows if r[2] is not None]
    fks = [r[3] for r in rows if r[3] is not None]
    fb = sum(1 for r in rows if r[4])
    ms = [r[5] for r in rows if r[5] is not None]
    ok = sum(
        1
        for r in rows
        if not r[4]
        and (r[0] or 0) >= 0.8
        and (r[1] or 0) >= 0.8
        and (r[2] or 0) >= 0.8
    )
    return {
        "n": len(rows),
        "fallback": fb,
        "pass_gate": ok,
        "f_avg": mean(fs) if fs else None,
        "ar_avg": mean(ars) if ars else None,
        "cp_avg": mean(cps) if cps else None,
        "fk_avg": mean(fks) if fks else None,
        "ms_avg": mean(ms) if ms else None,
    }


conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

cur.execute(
    """
    SELECT query_index, ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback,
           execution_time_ms, created_at, disease, query_level_label
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback,
               execution_time_ms, created_at, disease, query_level_label
        FROM public.rag_audit_log
        WHERE is_final = TRUE
          AND expected_tier = 0
          AND ablation_condition = 'C'
          AND query_index = ANY(%s)
        ORDER BY query_index, created_at DESC
    ) t
    ORDER BY query_index
    """,
    (FALLBACK_TRACK,),
)
tracked = cur.fetchall()

cur.execute(
    """
    SELECT query_index, ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback,
           execution_time_ms, created_at
    FROM (
        SELECT DISTINCT ON (query_index)
               query_index, ragas_f, ragas_ar, ragas_cp, fk_grade, is_fallback,
               execution_time_ms, created_at
        FROM public.rag_audit_log
        WHERE is_final = TRUE
          AND expected_tier = 0
          AND ablation_condition = 'C'
        ORDER BY query_index, created_at DESC
    ) t
    ORDER BY query_index
    """
)
all_c = cur.fetchall()

cur.execute(
    """
    SELECT MIN(created_at), MAX(created_at)
    FROM public.rag_audit_log
    WHERE is_final = TRUE
      AND ablation_condition = 'C'
      AND query_index = ANY(%s)
    """,
    (FALLBACK_TRACK,),
)
ts_range = cur.fetchone()

conn.close()

rows = [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in tracked]
all_rows = [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in all_c]

a = agg(rows)
b = agg(all_rows)

fb_idx = [r[0] for r in tracked if r[5]]

print("=== 조건 C 재실행 성능 (Supabase 최신 is_final per query_index) ===")
print(f"Fallback 추적 38건 로그 기간: {ts_range[0]} ~ {ts_range[1]}")
print()
print("[Fallback 추적 38건]")
print(f"  로그 수: {a['n']}/38")
print(f"  Fallback: {a['fallback']}")
print(f"  F/AR/CP 게이트 통과(비Fallback): {a['pass_gate']}/{a['n']}")
if a["f_avg"] is not None:
    print(f"  RAGAS 평균 — F={a['f_avg']:.3f} AR={a['ar_avg']:.3f} CP={a['cp_avg']:.3f}")
if a["fk_avg"] is not None:
    print(f"  FK 평균: {a['fk_avg']:.2f}")
if a["ms_avg"] is not None:
    print(f"  실행시간 평균: {a['ms_avg']/1000:.1f}s")
if fb_idx:
    print(f"  Fallback idx: {fb_idx}")

print()
print("[조건 C Tier0 전체 (최신 per idx)]")
print(f"  로그 수: {b['n']}")
print(f"  Fallback: {b['fallback']}")
print(f"  F/AR/CP 게이트 통과: {b['pass_gate']}/{b['n']}")
if b["f_avg"] is not None:
    print(f"  RAGAS 평균 — F={b['f_avg']:.3f} AR={b['ar_avg']:.3f} CP={b['cp_avg']:.3f}")

print()
print("=== 로컬 전체 재테스트 로그 요약 ===")
for path in [
    "test_fallback_final_38.txt",
    "test_fallback_final_38_v2.txt",
    "test_fallback_final_38_v3.txt",
    "test_fallback_final_38_v4.txt",
]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Fallback" in line and ("완료" in line or "?" in line):
                    print(f"  {path}: {line.strip()}")
                    break
    except FileNotFoundError:
        pass
