"""
재작성된 전체 122개 질문(Tier 1·2 에스컬레이션 대상 전체)에 대한 Tier 0 도달 여부 테스트.
Oracle INSERT 없이 파이프라인만 실행하여 final_tier를 확인한다.

사용법:
    python test_revised_questions.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch

# Oracle 저장 함수를 no-op으로 패치 (DB 오염 방지)
_noop = lambda *a, **kw: None

# 원래 Oracle Tier 1(63건) + Tier 2(59건) 에스컬레이션 인덱스 전체
ALL_ESCALATED_INDICES = {
    # ── Tier 1 출신 (63건) ──────────────────────────────────────
    1, 5, 6, 14, 15, 21, 34, 35, 39, 40, 47, 49, 60, 66,
    76, 77, 78, 81, 83, 84, 85, 87, 91, 97, 101, 102, 107,
    109, 111, 113, 114, 118, 136, 138, 140, 146, 148, 149,
    150, 151, 154, 158, 160, 167, 171, 175, 182, 195, 205,
    206, 213, 217, 221, 223, 224, 225, 226, 229, 230, 232,
    234, 238, 239,
    # ── Tier 2 출신 (59건) ──────────────────────────────────────
    7, 8, 9, 13, 31, 32, 37, 38, 44, 45, 48, 51, 54, 55,
    57, 63, 69, 72, 75, 89, 90, 92, 94, 98, 103, 108, 110,
    117, 125, 126, 127, 128, 134, 135, 141, 143, 159, 162,
    163, 170, 173, 176, 177, 181, 186, 188, 189, 197, 199,
    200, 201, 202, 211, 215, 227, 231, 235, 237, 240,
}

from stqs_questions import QUESTIONS

targets = [(disease, level, idx, text)
           for disease, level, idx, text in QUESTIONS
           if idx in ALL_ESCALATED_INDICES]

print("=" * 70)
print(f"  재작성 질문 Tier 테스트  ({len(targets)}건 / 원래 에스컬레이션 122건)")
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
            final_tier  = state.get("search_tier", -1)
            tier_path   = state.get("tier_path", "?")
            elapsed     = int((time.time() - t0) * 1000)
            ok = (final_tier == 0)
            tag = "OK " if ok else "ESC"
            print(f"  [{i:>3}/{len(targets)}] [{tag}] q{idx:>3} [{level}] "
                  f"tier={final_tier} path={tier_path} {elapsed}ms  {disease}")
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            print(f"  [{i:>3}/{len(targets)}] [ERR] q{idx:>3} [{level}] "
                  f"ERROR: {e}  {disease}")
            final_tier  = -1
            tier_path   = "error"
        results.append((idx, level, disease, final_tier, tier_path))

print()
print("=" * 70)
print("  테스트 결과 요약")
print("=" * 70)
total  = len(results)
tier0  = sum(1 for r in results if r[3] == 0)
tier1  = sum(1 for r in results if r[3] == 1)
tier2  = sum(1 for r in results if r[3] == 2)
errors = sum(1 for r in results if r[3] == -1)
print(f"  전체 {total}건  |  Tier 0: {tier0}건 ({tier0/total*100:.1f}%)  "
      f"|  Tier 1: {tier1}건  |  Tier 2: {tier2}건  |  오류: {errors}건")
print(f"  목표 (50% = {total // 2}건) 달성 여부: {'달성 ✓' if tier0 >= total * 0.5 else '미달성 ✗'}")
print()

if tier1 + tier2 > 0:
    print("  ▶ 여전히 에스컬레이션된 질문:")
    for idx, level, disease, tier, path in results:
        if tier not in (0, -1):
            text = next(q[3] for q in QUESTIONS if q[2] == idx)
            print(f"    q{idx:>3} [{level}] Tier{tier} path={path}  {disease}")
            print(f"         {text[:88]}{'...' if len(text) > 88 else ''}")
print("=" * 70)
