"""
전체 240개 질문 Tier 0 도달 여부 테스트.
Oracle INSERT 없이 파이프라인만 실행.
DB 저장 없음.
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch
_noop = lambda *a, **kw: None

from stqs_questions import QUESTIONS, STQS_EXPECTED_TOTAL

targets = sorted(QUESTIONS, key=lambda x: x[2])

assert len(targets) == STQS_EXPECTED_TOTAL, f"예상 {STQS_EXPECTED_TOTAL}건, 실제 {len(targets)}건"

print("=" * 70)
print(f"  전체 질문 Tier 테스트  ({len(targets)}건)  [DB 저장 없음]")
print("=" * 70)

with patch("infra.audit_logger.save_loop_log",  _noop), \
     patch("infra.audit_logger.save_audit_log", _noop):

    from graph import run_medical_self_corrective_rag

    results = []

    for i, (disease, level, idx, text) in enumerate(targets, 1):
        forced_level = "Professional" if level == "P" else "Consumer"
        t0 = time.time()
        try:
            state = run_medical_self_corrective_rag(
                question=text,
                forced_user_level=forced_level,
                ablation_condition="A",
                query_index=idx,
                disease=disease,
                query_level_label=level,
            )
            final_tier = state.get("search_tier", -1)
            tier_path  = state.get("tier_path", "?")
            elapsed    = int((time.time() - t0) * 1000)
            tag = "OK " if final_tier == 0 else "ESC"
            print(f"  [{i:>3}/{len(targets)}] [{tag}] q{idx:>3} [{level}] "
                  f"tier={final_tier} path={tier_path} {elapsed}ms  {disease}")
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            print(f"  [{i:>3}/{len(targets)}] [ERR] q{idx:>3} [{level}] "
                  f"ERROR: {e}  {disease}")
            final_tier = -1
            tier_path  = "error"
        results.append((idx, level, disease, final_tier, tier_path))
        sys.stdout.flush()

print()
print("=" * 70)
print("  전체 240건 결과 요약")
print("=" * 70)
total  = len(results)
tier0  = sum(1 for r in results if r[3] == 0)
tier1  = sum(1 for r in results if r[3] == 1)
tier2  = sum(1 for r in results if r[3] == 2)
errors = sum(1 for r in results if r[3] == -1)
print(f"  전체 {total}건  |  Tier 0: {tier0}건 ({tier0/total*100:.1f}%)"
      f"  |  Tier 1: {tier1}건  |  Tier 2: {tier2}건  |  오류: {errors}건")
print()

failing = [(idx, level, disease, tier, path) for idx, level, disease, tier, path in results if tier != 0]
if failing:
    print(f"  ▶ Tier 0 미달성 질문 ({len(failing)}건):")
    for idx, level, disease, tier, path in failing:
        text = next(q[3] for q in QUESTIONS if q[2] == idx)
        print(f"    q{idx:>3} [{level}] Tier{tier} path={path}  {disease}")
        print(f"         {text[:90]}{'...' if len(text) > 90 else ''}")
print("=" * 70)
