"""
Task1 — STQS-40 fallback/high-FK 질문 교체 (0-indexed 기준)

idx 6  (DB:7 )  뇌졸중 P   — rt-PA 기준 F=AR=CP=0.0  → CT/MRI 영상 역할
idx 12 (DB:13)  우울증 P   — DSM-5 F=AR=CP=0.0       → 주요 vs 지속성 우울장애
idx 24 (DB:25)  빈혈 P     — 헴 합성 AR=0.69          → 혈청 철·TIBC·페리틴 감별
idx 30 (DB:31)  CAP P      — CURB-65 AR=0.75          → 원인균별 항생제
idx 31 (DB:32)  CAP C      — FK=21.63                 → 감기 vs 폐렴 단순 감별
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

PATCHES = {
    # 뇌졸중 P: rt-PA(FAISS 미수록) → CT/MRI 영상 역할(FAISS 수록 확인)
    6:  ("급성 허혈성 뇌졸중의 초기 진단에서 비조영증강 CT와 확산강조 MRI(DWI)의 "
         "역할을 비교하고, 대혈관 폐색(large vessel occlusion) 확인을 위한 "
         "혈관 영상 검사(CTA/MRA)의 적응증을 기술하시오."),
    # 우울증 P: DSM-5(FAISS 미수록) → 주요 vs 지속성 우울장애 비교(FAISS 수록 확인)
    12: ("주요 우울장애와 지속성 우울장애(기분저하증, dysthymia)의 진단 기준상 "
         "최소 지속 기간 및 증상 심각도의 차이와, 각각에 대한 1차 선택 약물 치료를 비교하시오."),
    # 빈혈 P: 헴 합성 기전(AR 미달) → 혈청 검사 소견 감별(MSD 명확 수록)
    24: ("철결핍성 빈혈의 진단에서 혈청 철(serum iron), 총 철결합능(TIBC), "
         "혈청 페리틴(ferritin)의 특징적 변화 패턴과 이를 이용한 지중해성 빈혈 및 "
         "만성 질환 빈혈과의 감별 기준을 기술하시오."),
    # 지역사회획득폐렴 P: CURB-65(AR 미달) → 원인균별 항생제(MSD 명확 수록)
    30: ("지역사회획득폐렴의 주요 원인균(S. pneumoniae, Mycoplasma pneumoniae, "
         "Legionella pneumophila)별 임상적 특징과 각각에 대한 경험적 항생제 치료 권고를 기술하시오."),
    # 지역사회획득폐렴 C: 복잡 증상 목록(FK=21.63) → 단순 감별 질문
    31: "감기와 폐렴은 어떻게 구별하고, 어느 경우에 즉시 병원을 가야 하나요?",
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
        print(f"    구: {old_q[:70]!r}")
        print(f"    신: {new_q[:70]!r}")

target_cell['source'] = [new_src]

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ main.ipynb 저장 완료")
