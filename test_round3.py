"""
3차 재작성 84개 질문 Tier 0 도달 여부 테스트.
Oracle INSERT 없이 파이프라인만 실행.

대상:
  - 1차 세션 재작성 중 여전히 에스컬레이션된 37개 (first-batch)
  - 1차 세션 미테스트 12개 (first-batch, 429 오류)
  - 2차 세션 재작성 중 여전히 에스컬레이션된 35개 (new-revised)
  합계: 84개

사용법:
    python test_round3.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch
_noop = lambda *a, **kw: None

ROUND3_INDICES = {
    # ── 1차 세션 first-batch 에스컬레이션 37건 ─────────────────────────
    1, 6, 7, 14, 15, 21, 31, 32, 34, 35, 37, 38, 39, 44, 45, 48,
    51, 54, 55, 57, 63, 76, 77, 83, 84, 89, 90, 91, 92, 98, 102,
    103, 107, 108, 113, 118, 126,
    # ── 1차 세션 미테스트 12건 (429 오류) ──────────────────────────────
    134, 136, 138, 143, 148, 154, 159, 160, 162, 170, 173, 176,
    # ── 2차 세션 new-revised 에스컬레이션 35건 ──────────────────────────
    49, 69, 87, 97, 109, 117, 135, 140, 151, 158, 163, 175, 182,
    186, 188, 189, 197, 199, 200, 201, 206, 211, 217, 223, 224,
    225, 226, 229, 230, 231, 232, 234, 237, 238, 239,
}

from stqs_questions import QUESTIONS

targets = [(disease, level, idx, text)
           for disease, level, idx, text in QUESTIONS
           if idx in ROUND3_INDICES]

assert len(targets) == len(ROUND3_INDICES), f"매칭 실패: {len(targets)} != {len(ROUND3_INDICES)}"

print("=" * 70)
print(f"  3차 재작성 Tier 테스트  ({len(targets)}건)")
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
            print(f"  [{i:>2}/{len(targets)}] [{tag}] q{idx:>3} [{level}] "
                  f"tier={final_tier} path={tier_path} {elapsed}ms  {disease}")
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            print(f"  [{i:>2}/{len(targets)}] [ERR] q{idx:>3} [{level}] "
                  f"ERROR: {e}  {disease}")
            final_tier = -1
            tier_path  = "error"
        results.append((idx, level, disease, final_tier, tier_path))

print()
print("=" * 70)
print("  3차 재작성 결과 요약")
print("=" * 70)
total  = len(results)
tier0  = sum(1 for r in results if r[3] == 0)
tier1  = sum(1 for r in results if r[3] == 1)
tier2  = sum(1 for r in results if r[3] == 2)
errors = sum(1 for r in results if r[3] == -1)
print(f"  전체 {total}건  |  Tier 0: {tier0}건 ({tier0/total*100:.1f}%)"
      f"  |  Tier 1: {tier1}건  |  Tier 2: {tier2}건  |  오류: {errors}건")
print(f"  목표 (50% = {total//2}건) 달성 여부: {'달성 ✓' if tier0 >= total * 0.5 else '미달성 ✗'}")
print()

if tier1 + tier2 > 0:
    print("  ▶ 여전히 에스컬레이션된 질문:")
    for idx, level, disease, tier, path in results:
        if tier not in (0, -1):
            text = next(q[3] for q in QUESTIONS if q[2] == idx)
            print(f"    q{idx:>3} [{level}] Tier{tier} path={path}  {disease}")
            print(f"         {text[:90]}{'...' if len(text) > 90 else ''}")
print("=" * 70)
