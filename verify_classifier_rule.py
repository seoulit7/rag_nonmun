import sys; sys.stdout.reconfigure(encoding='utf-8')

from agents.classifier import _CONSUMER_RULE

# 오분류 6건 — 모두 Consumer로 잡혀야 함
consumer_tests = [
    ("COPD C",        "흡연이 만성 폐쇄성 폐질환(COPD)을 유발하는 이유는 무엇인가요?"),
    ("알츠하이머 C",  "알츠하이머병의 초기 기억력 저하는 나이 들면서 생기는 일반적인 건망증과 어떻게 다른가요?"),
    ("만성신장질환 C","만성 신장질환이 있으면 왜 빈혈이 생기나요?"),
    ("위식도역류 C",  "역류성 식도염을 오래 방치하면 왜 식도암이 생길 수 있나요?"),
    ("갑상선 C",      "갑상선 호르몬이 부족하면 왜 신진대사가 느려지나요?"),
    ("위궤양 C",      "위궤양 환자가 아스피린(aspirin)이나 이부프로펜(ibuprofen)을 피해야 하는 이유는 무엇인가요?"),
    # 신규 CAP C 질문도 확인
    ("CAP C (신)",    "폐렴에 걸리면 어떤 증상이 나타나며, 어느 경우에 즉시 병원을 방문해야 하나요?"),
    ("빈혈 C (신)",   "철분이 부족하면 왜 빈혈이 생기나요?"),
    ("알츠하이머 C(신)", "알츠하이머병의 주요 초기 증상에는 무엇이 있나요?"),
    ("만성신장 C(신)", "만성 신장질환 환자에게 빈혈이 흔한 이유는 무엇인가요?"),
]

# Professional 질문 — 잡히면 안 됨
professional_tests = [
    ("고혈압 P",   "본태성 고혈압에서 RAAS의 활성화 기전을 설명하고 ACEI와 ARB의 약리학적 작용 차이를 비교하시오."),
    ("당뇨 P",     "제2형 당뇨병의 병태생리학적 기전에서 인슐린 저항성이 고혈당 발생에 기여하는 과정을 설명하고, 메트포르민이 이 기전에 어떻게 작용하는지 약리학적 근거를 기술하시오."),
    ("뇌졸중 P",   "급성 허혈성 뇌졸중의 초기 진단에서 비조영증강 CT와 확산강조 MRI(DWI)의 역할을 비교하고, 대혈관 폐색(large vessel occlusion) 확인을 위한 혈관 영상 검사(CTA/MRA)의 적응증을 기술하시오."),
    ("COPD P",     "COPD에서 흡연이 기도 내 호중구(neutrophil) 침윤과 만성 염증을 유발하는 병태생리학적 기전을 설명하시오."),
    ("우울증 P",   "주요 우울장애와 지속성 우울장애(기분저하증, dysthymia)의 진단 기준상 최소 지속 기간 및 증상 심각도의 차이와, 각각에 대한 1차 선택 약물 치료를 비교하시오."),
    ("빈혈 P",     "철결핍성 빈혈의 진단에서 혈청 철(serum iron), 총 철결합능(TIBC), 혈청 페리틴(ferritin)의 특징적 변화 패턴과 이를 이용한 지중해성 빈혈 및 만성 질환 빈혈과의 감별 기준을 기술하시오."),
    ("CAP P",      "지역사회획득폐렴의 중증도 평가에 사용되는 CURB-65 점수 체계의 각 항목을 기술하시오."),
    ("고혈압 P(신)", "성인에서 본태성 고혈압을 진단하기 위한 수축기·이완기 혈압 수치 기준을 기술하시오."),
]

print("▶ Consumer 패턴 검출 (모두 True여야 함)")
all_ok = True
for name, q in consumer_tests:
    match = bool(_CONSUMER_RULE.search(q))
    status = "✓" if match else "✗ FAIL"
    if not match:
        all_ok = False
    print(f"  {status}  [{name}] {q[:60]}")

print()
print("▶ Professional 오탐 검사 (모두 False여야 함)")
for name, q in professional_tests:
    match = bool(_CONSUMER_RULE.search(q))
    status = "✗ OVERKILL" if match else "✓"
    if match:
        all_ok = False
    print(f"  {status}  [{name}] {q[:60]}")

print()
print("결과:", "전체 OK ✓" if all_ok else "일부 실패 — 패턴 조정 필요")
