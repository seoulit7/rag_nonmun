"""
expected_tier=0 AND is_fallback=TRUE 6건 질문 교체 (0-indexed 튜플 위치)

idx 6  (DB:7 )  뇌졸중 P   — AR=0.0 despite F=1.0 (FAISS 내용과 query 미스매치)
idx 12 (DB:13)  우울증 P   — F=AR=CP=0.0 (FAISS에 SSRI 기전 상세 내용 없음)
idx 27 (DB:28)  불안장애 C — AR=0.0 (MSD에 자율신경 신경생물학 내용 부족)
idx 30 (DB:31)  지역사회획득폐렴 P — F=AR=CP=0.0 (선천면역 세포 기전 미흡)
idx 31 (DB:32)  지역사회획득폐렴 C — F=0.33, AR=0.0
idx 34 (DB:35)  유방암 P   — F=AR=CP=0.0 (ER+ 증식 분자 기전 미흡)

idx 11 (DB:12)  알츠하이머병 C — 오분류(P)로 인한 fallback → Task3 분류기 수정으로 해결
idx 37 (DB:38)  위궤양 C        — 오분류(P)로 인한 fallback → Task3 분류기 수정으로 해결
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

PATCHES = {
    # 뇌졸중 P: 죽상반 파열 기전(FAISS 미흡) → rt-PA 치료 기준(MSD 명확히 수록)
    6:  ("급성 허혈성 뇌졸중에서 혈전용해제(rt-PA) 투여의 치료 시간 창, "
         "적응증, 절대적 금기증을 기술하시오."),
    # 우울증 P: SSRI 약동학(FAISS 미흡) → DSM-5 진단 기준(MSD 명확히 수록)
    12: ("주요 우울장애의 DSM-5 진단 기준에서 요구하는 핵심 증상 항목과 "
         "진단에 필요한 최소 지속 기간을 기술하시오."),
    # 불안장애 C: '왜 두근거리나요'(신경생물학 미흡) → '어떤 증상이 나타나나요'
    27: "불안장애가 있으면 어떤 증상이 나타나나요?",
    # 지역사회획득폐렴 P: 선천면역 세포 기전(FAISS 미흡) → CURB-65(MSD 명확히 수록)
    30: ("지역사회획득 폐렴의 중증도 평가에 사용되는 CURB-65 점수 체계의 "
         "각 항목과 점수별 권고 치료 장소(외래/입원/중환자실)를 기술하시오."),
    # 지역사회획득폐렴 C: '왜 고열·기침'(기전 설명 부족) → '어떤 증상, 언제 병원'
    31: "폐렴에 걸리면 어떤 증상이 나타나며, 어느 경우에 즉시 병원을 방문해야 하나요?",
    # 유방암 P: ER+ 증식 기전(FAISS 미흡) → 타목시펜 기전(MSD 명확히 수록)
    34: ("에스트로겐 수용체 양성(ER+) 유방암에서 타목시펜(tamoxifen)의 "
         "작용 기전과 주요 부작용을 기술하시오."),
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
        print(f"    신: {new_q!r}")

target_cell['source'] = [new_src]

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ main.ipynb 저장 완료")
