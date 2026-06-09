import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

PATCHES = {
    3: "제2형 당뇨병은 왜 생기나요?",
    5: "흡연이 심장 질환 위험을 높이는 이유는 무엇인가요?",
}

with open("main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

target_cell = None
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "QUESTIONS = [" in src:
        target_cell = cell
        break

src = "".join(target_cell["source"])
tuples_found = list(
    re.finditer(
        r'\(\s*"([^"]+)",\s*\n\s*"([PCpc])",\s*\n\s*(\d),\s*\n\s*"((?:[^"\\]|\\.)*)",\s*\n\s*\)',
        src,
        re.MULTILINE,
    )
)

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
    print(f"  [tuple {i}/C] 신: {new_q}")

target_cell["source"] = [new_src]
with open("main.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✓ main.ipynb Consumer FK 질문 패치 완료")
