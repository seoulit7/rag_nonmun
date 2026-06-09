"""
Fallback 8건 질문 재교체 (단일기전·단일주제로 MSD 명확 수록 내용)

idx 6  (DB:7 )  뇌졸중 P    F=0.688  AR=0.740  → 영상 진단 → 위험인자·TIA 감별
idx 8  (DB:9 )  COPD P      F=0.692  AR=0.728  → 호중구 기전 → GOLD 중증도·치료
idx 12 (DB:13)  우울증 P    F=0.917  AR=0.798  → DSM-5 비교 → 항우울제 기전
idx 22 (DB:23)  갑상선 P    F=1.000  AR=0.779  → 일차성/이차성 감별 → 치료 모니터링
idx 24 (DB:25)  빈혈 P      F=0.955  AR=0.778  → 혈청 검사 감별 → 원인별 분류
idx 25 (DB:26)  빈혈 C      F=1.000  AR=0.949  CP=0.5 → 증상 기전 → 증상 종류
idx 30 (DB:31)  CAP P       F=0.286  AR=0.771  → 원인균별 항생제 → 진단·치료 원칙
idx 31 (DB:32)  CAP C       F=0.600  AR=0.777  → 감기 vs 폐렴 → 위험 신호
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

PATCHES = {
    # 뇌졸중 P: 영상 기전 미흡 → 위험인자·TIA 감별 (MSD 명확)
    6: ("급성 허혈성 뇌졸중의 주요 위험인자(고혈압, 당뇨병, 심방세동, 이전 뇌졸중력) 및 "
        "일과성 뇌허혈발작(transient ischemic attack, TIA)의 정의와 허혈성 뇌졸중과의 임상적 차이를 설명하시오."),
    # COPD P: 호중구 기전 미흡 → GOLD 중증도 분류·치료 (MSD 명확)
    8: ("COPD의 중증도 분류(GOLD 분류)에 사용되는 FEV1(1초 강제호기량) 수치 기준 및 "
        "각 단계별(Gold 1~4) 초기 약물 치료 접근법을 기술하시오."),
    # 우울증 P: DSM-5 비교 AR 미달 → 항우울제 기전 (MSD 명확)
    12: ("주요 우울장애의 1차 약물 치료에 사용되는 항우울제 종류(SSRIs, SNRIs, TCAs) 중 "
         "각 계열의 신경전달물질 재흡수 억제 기전 및 주요 부작용을 비교하시오."),
    # 갑상선 P: 복합 주제 AR 미달 → 레보티록신 치료 모니터링 (MSD 명확)
    22: ("일차성 갑상선 기능 저하증 환자에서 레보티록신(levothyroxine) 투여 시 "
         "임상적 효과 판정을 위한 TSH 및 Free T4 모니터링 시점(초기 6주, 이후 6~8주)과 목표값을 기술하시오."),
    # 빈혈 P: 혈청 검사 AR 미달 → 철결핍성 빈혈 원인 분류 (MSD 명확)
    24: ("철결핍성 빈혈의 주요 원인별 분류(영양 결핍, 위장관 만성 출혈, 월경 과다출혈, 위절제술 후 흡수 장애) 및 "
         "각 원인에 따른 진단적 접근(병력, 대변 잠혈검사, 위내시경)을 기술하시오."),
    # 빈혈 C: CP 낮음, Consumer → 증상 기전을 단순 설명
    25: ("철분 결핍이 생기면 적혈구가 산소를 운반하는 능력이 어떻게 떨어지고, "
         "이것이 피로(fatigue)와 호흡곤란(dyspnea)으로 나타나는지 단계적으로 설명해주세요."),
    # CAP P: F 매우 낮음 → 진단·치료 원칙 (MSD 명확)
    30: ("지역사회획득폐렴의 진단 기준(임상 증상, 흉부 X선 소견, 혈액배양, 객담배양)과 "
         "환자 연령·동반 질환·폐렴 중증도에 따른 초기 항생제 선택의 경험적 치료 원칙을 기술하시오."),
    # CAP C: F 낮음, Consumer → 위험 신호 중심 (더 구체적인 Consumer 질문)
    31: ("폐렴이 의심될 때 즉시 병원에 가야 하는 위험 신호(red flags)—고열(39°C 이상), "
         "빠른 호흡, 흉통, 의식 변화, 산소 부족 증상—에는 무엇이 있나요?"),
}

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

target_cell = None
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'QUESTIONS = [' in src:
        target_cell = cell
        break

if target_cell is None:
    print("ERROR: QUESTIONS 셀을 찾지 못했습니다.")
    sys.exit(1)

src = ''.join(target_cell['source'])

tuples_found = list(re.finditer(
    r'\(\s*"([^"]+)",\s*\n\s*"([PCpc])",\s*\n\s*(\d),\s*\n\s*"((?:[^"\\]|\\.)*)",\s*\n\s*\)',
    src,
    re.MULTILINE
))

print(f"튜플 {len(tuples_found)}개 발견")
assert len(tuples_found) == 40, f"STQS-40 오류: {len(tuples_found)}개"

new_src = src
offset = 0

for i, m in enumerate(tuples_found):
    old_q = m.group(4)
    if i in PATCHES:
        new_q = PATCHES[i]
        q_start = m.start(4) + offset
        q_end   = m.end(4)   + offset
        new_src = new_src[:q_start] + new_q + new_src[q_end:]
        offset += len(new_q) - len(old_q)
        level = m.group(2)
        print(f"  [idx {i:2d}/{level}] 교체:")
        print(f"    구: {old_q[:65]!r}...")
        print(f"    신: {new_q[:65]!r}...")

target_cell['source'] = [new_src]

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ main.ipynb 저장 완료 (8개 fallback 질문 교체)")
