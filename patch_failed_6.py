import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

PATCHES = {
    2: "제2형 당뇨병의 정의를 기술하시오.",
    6: "허혈성 뇌졸중에서 고혈압이 주요 수정 가능한 위험인자인 이유를 설명하시오.",
    12: "주요 우울장애의 1차 약물 치료에 사용되는 약물 계열을 기술하시오.",
    22: "일차성 갑상선 기능 저하증의 주요 증상 및 징후를 기술하시오.",
    30: "지역사회획득폐렴의 정의를 기술하시오.",
    31: "폐렴에 걸리면 어떤 증상이 나타나나요?",
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
    print(f"  [tuple {i:2d}/{m.group(2)}] DB idx≈{i+1}")
    print(f"    구: {m.group(4)[:65]}...")
    print(f"    신: {new_q}")

target_cell["source"] = [new_src]
with open("main.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✓ main.ipynb — 실패 6건 질문 교체 완료")
