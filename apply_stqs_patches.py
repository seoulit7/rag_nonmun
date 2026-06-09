import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

PATCHES = {
    0: "성인에서 본태성 고혈압을 진단하기 위한 수축기·이완기 혈압 수치 기준을 기술하시오.",
    2: "제2형 당뇨병의 진단에 사용되는 HbA1c 수치 기준을 기술하시오.",
    6: "급성 허혈성 뇌졸중에서 혈전용해제(rt-PA) 투여의 치료 시간 창과 절대적 금기증을 기술하시오.",
    8: "COPD의 중증도 분류(GOLD 분류)에 사용되는 FEV1(1초 강제호기량) 수치 기준을 기술하시오.",
    11: "알츠하이머병의 주요 초기 증상에는 무엇이 있나요?",
    12: "주요 우울장애의 DSM-5 진단 기준에서 요구하는 핵심 증상 항목과 진단에 필요한 최소 지속 기간을 기술하시오.",
    19: "만성 신장질환 환자에게 빈혈이 흔한 이유는 무엇인가요?",
    22: "일차성 갑상선 기능 저하증의 가장 흔한 원인을 기술하시오.",
    25: "철분이 부족하면 왜 빈혈이 생기나요?",
    30: "지역사회획득폐렴의 중증도 평가에 사용되는 CURB-65 점수 체계의 각 항목을 기술하시오.",
    31: "폐렴에 걸리면 어떤 증상이 나타나며, 어느 경우에 즉시 병원을 방문해야 하나요?",
}

with open("main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

target_cell = None
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "QUESTIONS = [" in src:
        target_cell = cell
        break

if target_cell is None:
    print("ERROR: QUESTIONS 셀 없음")
    sys.exit(1)

src = "".join(target_cell["source"])
tuples_found = list(
    re.finditer(
        r'\(\s*"([^"]+)",\s*\n\s*"([PCpc])",\s*\n\s*(\d),\s*\n\s*"((?:[^"\\]|\\.)*)",\s*\n\s*\)',
        src,
        re.MULTILINE,
    )
)
assert len(tuples_found) == 40

new_src = src
offset = 0
for i, m in enumerate(tuples_found):
    if i not in PATCHES:
        continue
    new_q = PATCHES[i]
    q_start = m.start(4) + offset
    q_end = m.end(4) + offset
    new_src = new_src[:q_start] + new_q + new_src[q_end:]
    offset += len(new_q) - len(m.group(4))
    print(f"  [tuple {i:2d}/{m.group(2)}] DB query_index≈{i+1}")
    print(f"    신: {new_q}")

target_cell["source"] = [new_src]
with open("main.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ main.ipynb — {len(PATCHES)}개 질문 교체 완료")
