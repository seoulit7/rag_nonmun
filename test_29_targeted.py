"""
29개 재작성 질문만 타겟 테스트 (DB 조회 없이 고정 인덱스 사용)
- Oracle DB WRITE 없음 (audit_logger 패치)
- 결과를 JSON 파일로 저장
"""
import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch
_noop = lambda *a, **kw: None

sys.path.insert(0, r"c:\dev\rag_ai\rag_nonmun")

TARGET_INDICES = [
    26, 34, 38, 41, 43, 44, 46, 48, 56, 58, 59, 60,
    77, 97, 98, 99, 117, 141, 145, 158, 163, 179,
    188, 211, 222, 228, 232, 234, 239
]

from stqs_questions import QUESTIONS
q_map = {i: (d, l, q) for d, l, i, q in QUESTIONS}

TARGET_QS = [(qi, *q_map[qi]) for qi in TARGET_INDICES if qi in q_map]

print("=" * 70)
print(f"  [타겟 테스트] {len(TARGET_QS)}개 질문 파이프라인 실행 (DB 저장 없음)")
print("=" * 70)
for qi, d, l, q in TARGET_QS:
    print(f"  q{qi:>3} [{l}] {d}  {q[:60]}...")
print()
sys.stdout.flush()

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

            tag = "PASS" if final_tier == 0 else "FAIL"
            print(f"  [{i:>2}/{len(TARGET_QS)}] [{tag}] q{qi:>3} [{level}] "
                  f"Tier {final_tier}  path={tier_path}  "
                  f"{elapsed//1000}s  {disease}")
            sys.stdout.flush()

        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            print(f"  [{i:>2}/{len(TARGET_QS)}] [ERR ] q{qi:>3} [{level}] ERROR: {e}  {disease}")
            final_tier = -1
            tier_path  = "error"
            ragas_f = ragas_ar = ragas_cp = None
            sys.stdout.flush()

        results.append({
            "query_index": qi,
            "disease": disease,
            "level": level,
            "question": text[:200],
            "new_tier": final_tier,
            "new_path": tier_path,
            "new_f":  float(ragas_f)  if ragas_f  is not None else None,
            "new_ar": float(ragas_ar) if ragas_ar is not None else None,
            "new_cp": float(ragas_cp) if ragas_cp is not None else None,
            "pass": (final_tier == 0),
            "elapsed_ms": elapsed,
        })

total_elapsed = int(time.time() - total_start)

out_path = r"c:\dev\rag_ai\rag_nonmun\test_29_targeted_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_tested": len(results),
        "total_elapsed_sec": total_elapsed,
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print()
print("=" * 70)
print("  결과 요약")
print("=" * 70)

n_total = len(results)
n_pass  = sum(1 for r in results if r["new_tier"] == 0)
n_tier1 = sum(1 for r in results if r["new_tier"] == 1)
n_tier2 = sum(1 for r in results if r["new_tier"] == 2)
n_err   = sum(1 for r in results if r["new_tier"] == -1)

print(f"  테스트 대상: {n_total}개")
print(f"  Tier 0 달성 (PASS): {n_pass}개  ({n_pass/n_total*100:.1f}%)")
print(f"  Tier 1 에스컬레이션: {n_tier1}개")
print(f"  Tier 2 에스컬레이션: {n_tier2}개")
print(f"  오류: {n_err}개")
print(f"  총 소요: {total_elapsed//60}분 {total_elapsed%60}초")

still_fail = [r for r in results if r["new_tier"] != 0]
if still_fail:
    print(f"\n  ▶ 여전히 실패 ({len(still_fail)}개):")
    for r in still_fail:
        print(f"    q{r['query_index']:>3} [{r['level']}] Tier {r['new_tier']}  {r['disease']}")

print()
print(f"  상세 결과 저장: {out_path}")
print("=" * 70)
