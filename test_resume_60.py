"""
test_new_60_questions.py 중단 이후 재개 스크립트.
중단 지점: q186 (Migraine-C) 처리 중 → q175부터 안전하게 재시작.

사용법:
    python test_resume_60.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch

_noop = lambda *a, **kw: None

# q175 이후 미완료 35개 (q175 포함, 안전 마진 적용)
RESUME_INDICES = {
    175,  # Lung-P: 담배 발암물질/DNA 손상
    177,  # Lung-P: 저선량CT vs 흉부X선
    181,  # Migraine-P: 피질 확산 억제/시각 전조
    182,  # Migraine-P: 프로프라놀롤 기전
    186,  # Migraine-C: 편두통 약 타이밍
    188,  # PID-P: 나팔관 유착/불임
    189,  # PID-P: NAAT vs 배양
    195,  # Prostate-P: ADT/테스토스테론 기전
    197,  # Prostate-C: 위험인자
    199,  # Psoriasis-P: T세포/사이토카인 축
    200,  # Psoriasis-P: 바이오로직 vs 면역억제
    201,  # Psoriasis-P: 건선관절염 30%
    202,  # Psoriasis-C: 건선 외양
    205,  # PedT2D-P: 비만/인슐린저항
    206,  # PedT2D-P: 메트포르민 기전
    211,  # Urticaria-P: 히스타민/팽진 기전
    213,  # Urticaria-P: H1 항히스타민 기전
    215,  # Urticaria-C: 유발 인자
    217,  # UF-P: 에스트로겐/근종 성장
    221,  # UF-C: 폐경 후 자연 축소
    223,  # Dyslip-P: LDL/동맥 플라크 형성
    224,  # Dyslip-P: 스타틴/간 LDL 저하
    225,  # Dyslip-P: HDL 역콜레스테롤 수송
    226,  # Dyslip-C: LDL이 나쁜 이유
    227,  # Dyslip-C: 식이 변화
    229,  # HF-P: RAAS/나트륨 저류
    230,  # HF-P: BNP/NT-proBNP 감별
    231,  # HF-P: ACE 억제제/리모델링
    232,  # HF-C: orthopnea
    234,  # HF-C: 체중 2kg 경보
    235,  # VV-P: 판막 기능부전 기전
    237,  # VV-P: 압박스타킹 기전
    238,  # VV-C: 다리 무거움/붓기
    239,  # VV-C: 장시간 부동 악화
    240,  # VV-C: 최소침습 치료
}

from stqs_questions import QUESTIONS

targets = [(disease, level, idx, text)
           for disease, level, idx, text in QUESTIONS
           if idx in RESUME_INDICES]

print("=" * 70)
print(f"  재개 테스트  ({len(targets)}건 / q175~q240)")
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
            ok  = (final_tier == 0)
            tag = "OK " if ok else "ESC"
            print(f"  [{i:>2}/{len(targets)}] [{tag}] q{idx:>3} [{level}] "
                  f"tier={final_tier} path={tier_path} {elapsed}ms  {disease}")
        except Exception as e:
            elapsed    = int((time.time() - t0) * 1000)
            print(f"  [{i:>2}/{len(targets)}] [ERR] q{idx:>3} [{level}] "
                  f"ERROR: {e}  {disease}")
            final_tier = -1
            tier_path  = "error"
        results.append((idx, level, disease, final_tier, tier_path))

print()
print("=" * 70)
print("  재개 구간 결과 요약")
print("=" * 70)
total  = len(results)
tier0  = sum(1 for r in results if r[3] == 0)
tier1  = sum(1 for r in results if r[3] == 1)
tier2  = sum(1 for r in results if r[3] == 2)
errors = sum(1 for r in results if r[3] == -1)
print(f"  전체 {total}건  |  Tier 0: {tier0}건 ({tier0/total*100:.1f}%)  "
      f"|  Tier 1: {tier1}건  |  Tier 2: {tier2}건  |  오류: {errors}건")
print()

if tier1 + tier2 > 0:
    print("  ▶ 여전히 에스컬레이션된 질문:")
    for idx, level, disease, tier, path in results:
        if tier not in (0, -1):
            text = next(q[3] for q in QUESTIONS if q[2] == idx)
            print(f"    q{idx:>3} [{level}] Tier{tier} path={path}  {disease}")
            print(f"         {text[:88]}{'...' if len(text) > 88 else ''}")
print("=" * 70)
