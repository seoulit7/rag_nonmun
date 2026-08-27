"""
이번 세션에서 새로 재작성된 59개 질문(기존 분자기전 수준 → MSD Manual 임상 수준)
Tier 0 도달 여부 테스트. Oracle INSERT 없이 파이프라인만 실행.

사용법:
    python test_new_60_questions.py
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from unittest.mock import patch

_noop = lambda *a, **kw: None

# 이번 세션에서 새로 재작성된 59개 (이전에 "분자기전 수준" 이유로 보류했던 것들)
NEW_REVISED_INDICES = {
    # ── Tier 1 출신 미처리 32건 ───────────────────────────────────────────
    49,   # OA-P: MMP 효소 → 염증과 연골 파괴 관계
    85,   # Osteo-P: RANK/RANKL/OPG → 조골/파골세포 균형
    87,   # Osteo-P: 에스트로겐/파골세포 → 폐경 후 골소실
    97,   # CRC-P: 좌측 종양 혈변 해부학적 이유
    109,  # PUD-P: NSAID 전신 기전 → 위 방어 손상
    111,  # PUD-P: 프로스타글란딘/COX → 위 점막 보호 기능
    140,  # Chol-P: 담관 폐쇄 빌리루빈 → 담도산통 기전
    146,  # CP-P: 알코올/췌장 선세포 손상 기전
    149,  # CP-C: 만성 췌장염 주요 원인 물질
    150,  # CP-C: 음주·흡연이 만성 췌장염 악화시키는 이유
    151,  # Endo-P: 나팔관 유착/불임 기전
    158,  # HIV-P: RT 오류/병합요법 필요성
    167,  # IBS-C: 장-뇌 축 / 기여 요인
    171,  # LSS-P: 퇴행성 구조 변화 → 척추관 협착
    175,  # Lung-P: 담배 발암물질/DNA 손상 기전
    182,  # Migraine-P: 프로프라놀롤/베타차단 기전
    195,  # Prostate-P: ADT/테스토스테론 기전
    205,  # PedT2D-P: 비만/인슐린저항 기전
    206,  # PedT2D-P: 메트포르민 대사 작용
    213,  # Urticaria-P: H1 항히스타민 수용체 기전
    217,  # UF-P: 에스트로겐/평활근 증식 기전
    221,  # UF-C: 폐경 후 자궁근종 자연 축소 이유
    223,  # Dyslip-P: LDL/동맥 내막 분자 이벤트
    224,  # Dyslip-P: HMG-CoA/LDL 수용체 하류 효과
    225,  # Dyslip-P: HDL 역콜레스테롤 수송
    226,  # Dyslip-C: LDL이 나쁜 이유
    229,  # HF-P: RAAS/알도스테론/나트륨 재흡수
    230,  # HF-P: BNP/NT-proBNP 심장 vs 폐 감별
    232,  # HF-C: 누울 때 호흡곤란 (orthopnea)
    234,  # HF-C: 체중 2kg 증가 경보 의미
    238,  # VV-C: 종일 다리가 무겁고 아픈 이유
    239,  # VV-C: 장시간 부동이 정맥 혈액 정체 악화
    # ── Tier 2 출신 미처리 27건 ───────────────────────────────────────────
    8,    # T2D-P: 인슐린 수용체 신호 결함
    69,   # Hypo-P: 레보티록신 흡수 저해 물질
    75,   # Anemia-P: 경구 철 부작용 줄이는 복용 수정
    110,  # PUD-P: H.pylori 점막 방어 파괴
    117,  # AS-P: TNF 억제제 기전
    127,  # App-P: 맹장염 폐쇄 → 세균 과증식/혈관 손상
    128,  # App-P: 천공 합병증
    135,  # COVID-P: 항바이러스제 조기 투여 이유
    141,  # Chol-P: 무증상 담석 경과 관찰 근거
    163,  # IBS-P: 내장 과민성/점막 팽창 민감
    177,  # Lung-P: 저선량 CT vs 흉부 X선 비교
    181,  # Migraine-P: 피질 확산 억제 / 시각 전조
    186,  # Migraine-C: 편두통 약 타이밍 원칙
    188,  # PID-P: 나팔관 유착/불임 기전
    189,  # PID-P: NAAT vs 배양 진단 우위
    197,  # Prostate-C: 위험 인자 (연령·인종·가족력)
    199,  # Psoriasis-P: T세포/사이토카인 축
    200,  # Psoriasis-P: 바이오로직 vs 비선택적 면역억제
    201,  # Psoriasis-P: 건선관절염 30% 임상 함의
    202,  # Psoriasis-C: 건선 피부 병변 외양
    211,  # Urticaria-P: 히스타민/팽진 혈관 기전
    215,  # Urticaria-C: 두드러기 유발 인자
    227,  # Dyslip-C: LDL 낮추는 식이 변화
    231,  # HF-P: ACE 억제제/심장 리모델링 신경호르몬
    235,  # VV-P: 판막 기능 부전 기전
    237,  # VV-P: 압박스타킹 기전
    240,  # VV-C: 최소 침습 치료 옵션
}

from stqs_questions import QUESTIONS

targets = [(disease, level, idx, text)
           for disease, level, idx, text in QUESTIONS
           if idx in NEW_REVISED_INDICES]

print("=" * 70)
print(f"  신규 재작성 59개 질문 Tier 테스트  ({len(targets)}건)")
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
print("  테스트 결과 요약")
print("=" * 70)
total  = len(results)
tier0  = sum(1 for r in results if r[3] == 0)
tier1  = sum(1 for r in results if r[3] == 1)
tier2  = sum(1 for r in results if r[3] == 2)
errors = sum(1 for r in results if r[3] == -1)
print(f"  전체 {total}건  |  Tier 0: {tier0}건 ({tier0/total*100:.1f}%)  "
      f"|  Tier 1: {tier1}건  |  Tier 2: {tier2}건  |  오류: {errors}건")
print(f"  목표 (50% = {total//2}건) 달성 여부: {'달성 ✓' if tier0 >= total * 0.5 else '미달성 ✗'}")
print()

if tier1 + tier2 > 0:
    print("  ▶ 여전히 에스컬레이션된 질문:")
    for idx, level, disease, tier, path in results:
        if tier not in (0, -1):
            text = next(q[3] for q in QUESTIONS if q[2] == idx)
            print(f"    q{idx:>3} [{level}] Tier{tier} path={path}  {disease}")
            print(f"         {text[:88]}{'...' if len(text) > 88 else ''}")
print("=" * 70)
