"""
Oracle DB에서 조건 A의 Tier 0 실패 질문을 조회 →
재작성된 질문으로 파이프라인 실행 → Tier 0 달성 여부 확인.

- Oracle DB WRITE 없음 (audit_logger 패치)
- 결과를 JSON 파일로 저장
- 직전 DB 결과와 개선 현황 비교 출력
"""
import sys, io, time, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch
_noop = lambda *a, **kw: None

# ── 1. Oracle DB에서 Tier 0 실패 목록 조회 ───────────────────────────────
import sys
sys.path.insert(0, r"c:\dev\rag_ai\rag_nonmun")

import oracledb
import config.settings as settings

print("=" * 70)
print("  [STEP 1] Oracle DB에서 조건 A Tier 0 실패 질문 조회")
print("=" * 70)

conn = oracledb.connect(
    user=settings.ORACLE_USER,
    password=settings.ORACLE_PASSWORD,
    dsn=settings.ORACLE_DSN,
)
cur = conn.cursor()

cur.execute("""
    SELECT a.query_index, a.disease, a.query_level_label,
           a.final_tier, a.tier_path,
           a.ragas_f, a.ragas_ar, a.ragas_cp, a.q_total
    FROM rag_audit_log a
    INNER JOIN (
        SELECT query_index, MAX(ROWID) AS max_rid
        FROM rag_audit_log
        WHERE is_final = 1
          AND ablation_condition = 'A'
          AND query_index IS NOT NULL
        GROUP BY query_index
    ) b ON a.query_index = b.query_index AND a.ROWID = b.max_rid
    WHERE a.final_tier != 0
    ORDER BY a.query_index
""")
db_fails = cur.fetchall()
cur.close()
conn.close()

# DB 기준 실패 정보 저장
DB_FAIL_INFO = {}
for row in db_fails:
    qi, disease, level, tier, path, f, ar, cp, qt = row
    DB_FAIL_INFO[qi] = {
        "disease": disease, "level": level,
        "old_tier": tier, "old_path": path,
        "old_f": float(f) if f is not None else None,
        "old_ar": float(ar) if ar is not None else None,
        "old_cp": float(cp) if cp is not None else None,
        "old_qt": float(qt) if qt is not None else None,
    }

FAIL_INDICES = sorted(DB_FAIL_INFO.keys())
print(f"  DB 기준 Tier 0 실패: {len(FAIL_INDICES)}개")
print(f"  대상 query_index: {FAIL_INDICES}")
print()

# ── 2. 해당 질문 텍스트 로드 ─────────────────────────────────────────────
from stqs_questions import QUESTIONS
q_map = {i: (d, l, q) for d, l, i, q in QUESTIONS}

# DB 실패 목록 중 현재 questions.py에 없는 인덱스 확인
missing = [i for i in FAIL_INDICES if i not in q_map]
if missing:
    print(f"  경고: query_index {missing}가 QUESTIONS에 없음")

TARGET_QS = [(qi, *q_map[qi]) for qi in FAIL_INDICES if qi in q_map]
print(f"  테스트 대상: {len(TARGET_QS)}개 질문")
print()

# ── 3. 파이프라인 실행 ────────────────────────────────────────────────────
print("=" * 70)
print(f"  [STEP 2] {len(TARGET_QS)}개 질문 파이프라인 실행 (DB 저장 없음)")
print("=" * 70)

with patch("infra.audit_logger.save_loop_log",  _noop), \
     patch("infra.audit_logger.save_audit_log", _noop):

    from graph import run_medical_self_corrective_rag

    results = []
    total_start = time.time()

    for i, (qi, disease, level, text) in enumerate(TARGET_QS, 1):
        forced_level = "Professional" if level == "P" else "Consumer"
        t0 = time.time()
        try:
            state = run_medical_self_corrective_rag(
                question=text,
                forced_user_level=forced_level,
                ablation_condition="A",
                query_index=qi,
                disease=disease,
                query_level_label=level,
            )
            final_tier = state.get("search_tier", -1)
            tier_path  = state.get("tier_path", "?")
            ragas_f    = state.get("critic_score")
            ragas_ar   = state.get("answer_relevance_score")
            ragas_cp   = state.get("context_precision_score")
            elapsed    = int((time.time() - t0) * 1000)

            old_tier = DB_FAIL_INFO[qi]["old_tier"]
            improved = (final_tier == 0)
            tag = "PASS" if improved else "FAIL"

            print(f"  [{i:>2}/{len(TARGET_QS)}] [{tag}] q{qi:>3} [{level}] "
                  f"Tier {old_tier}→{final_tier}  path={tier_path}  "
                  f"{elapsed//1000}s  {disease}")
            sys.stdout.flush()

        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            print(f"  [{i:>2}/{len(TARGET_QS)}] [ERR ] q{qi:>3} [{level}] "
                  f"ERROR: {e}  {disease}")
            final_tier = -1
            tier_path  = "error"
            ragas_f = ragas_ar = ragas_cp = None
            sys.stdout.flush()

        results.append({
            "query_index": qi,
            "disease": disease,
            "level": level,
            "question": text[:200],
            "old_tier": DB_FAIL_INFO[qi]["old_tier"],
            "old_path": DB_FAIL_INFO[qi]["old_path"],
            "new_tier": final_tier,
            "new_path": tier_path,
            "new_f":  float(ragas_f)  if ragas_f  is not None else None,
            "new_ar": float(ragas_ar) if ragas_ar is not None else None,
            "new_cp": float(ragas_cp) if ragas_cp is not None else None,
            "improved": (final_tier == 0),
            "elapsed_ms": elapsed,
        })

total_elapsed = int(time.time() - total_start)

# ── 4. 결과 저장 ──────────────────────────────────────────────────────────
out_path = r"c:\dev\rag_ai\rag_nonmun\test_tier0_tuned_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_tested": len(results),
        "total_elapsed_sec": total_elapsed,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

# ── 5. 최종 요약 ─────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  결과 요약")
print("=" * 70)

n_total   = len(results)
n_pass    = sum(1 for r in results if r["new_tier"] == 0)
n_tier1   = sum(1 for r in results if r["new_tier"] == 1)
n_tier2   = sum(1 for r in results if r["new_tier"] == 2)
n_err     = sum(1 for r in results if r["new_tier"] == -1)

print(f"  테스트 대상: {n_total}개  (직전 DB Tier 0 실패 목록)")
print(f"  Tier 0 달성 (PASS): {n_pass}개  ({n_pass/n_total*100:.1f}%)")
print(f"  Tier 1 에스컬레이션: {n_tier1}개")
print(f"  Tier 2 에스컬레이션: {n_tier2}개")
print(f"  오류: {n_err}개")
print(f"  총 소요: {total_elapsed//60}분 {total_elapsed%60}초")
print()

still_fail = [r for r in results if r["new_tier"] != 0]
if still_fail:
    print(f"  ▶ 여전히 Tier 0 미달 ({len(still_fail)}개):")
    for r in still_fail:
        print(f"    q{r['query_index']:>3} [{r['level']}] "
              f"Tier {r['old_tier']}→{r['new_tier']}  {r['disease']}")

print()
print(f"  상세 결과 저장: {out_path}")
print("=" * 70)
