"""
4차 재작성 96개 질문 Tier 0 도달 여부 테스트.
Oracle INSERT 없이 파이프라인만 실행.

대상: 조건 A ablation 실행에서 Tier 0 미달성한 96개 질문 (query_index 확정 목록)
목표: 96개 중 50% 이상(48개) Tier 0 도달

사용법:
    python test_round4.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch
_noop = lambda *a, **kw: None

ROUND4_INDICES = {
    # Hypertension
    1, 3, 5, 6,
    # Type 2 Diabetes
    7,
    # Coronary Artery Disease
    15, 16, 17, 18,
    # Stroke
    20,
    # COPD
    26,
    # Alzheimer Disease
    31, 32, 34,
    # Major Depressive Disorder
    37, 38, 39, 40, 42,
    # Asthma
    43, 44, 45,
    # Osteoarthritis
    49, 51,
    # Chronic Kidney Disease
    55, 56, 57, 58, 60,
    # GERD
    63, 66,
    # Generalized Anxiety Disorder
    83, 84,
    # Osteoporosis
    87, 88, 89, 90,
    # Community-Acquired Pneumonia
    91, 92, 94,
    # Colorectal Cancer
    97, 98, 102,
    # Breast Cancer
    103, 107, 108,
    # Peptic Ulcer Disease
    114,
    # Ankylosing Spondylitis
    117, 118, 120,
    # Aplastic Anemia
    125, 126,
    # Appendicitis
    127, 128,
    # COVID-19
    133,
    # Cholelithiasis
    140, 141,
    # Chronic Pancreatitis
    145, 150,
    # Endometriosis
    154,
    # HIV Infection
    157, 158, 159, 160, 162,
    # Irritable Bowel Syndrome
    163, 164, 167,
    # Lumbar Spinal Stenosis
    170,
    # Lung Cancer
    175, 179,
    # Migraine
    181, 182, 184,
    # Pelvic Inflammatory Disease
    187,
    # Prostate Cancer
    194, 197, 198,
    # Psoriasis
    199, 200,
    # Pediatric Type 2 Diabetes
    205, 206,
    # Urticaria
    211, 213, 214,
    # Uterine Fibroid
    221,
    # Dyslipidemia
    224, 225, 226, 228,
    # Heart Failure
    230, 231, 232,
    # Varicose Veins
    235, 237, 239,
}

assert len(ROUND4_INDICES) == 96, f"Expected 96, got {len(ROUND4_INDICES)}"

from stqs_questions import QUESTIONS

targets = [(disease, level, idx, text)
           for disease, level, idx, text in QUESTIONS
           if idx in ROUND4_INDICES]

assert len(targets) == 96, f"매칭 실패: {len(targets)} != 96"

targets.sort(key=lambda x: x[2])

print("=" * 70)
print(f"  4차 재작성 Tier 테스트  ({len(targets)}건)")
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
print("  4차 재작성 결과 요약")
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
