import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

PATCHES = {
    1: "고혈압은 왜 생기나요?",
    18: "만성 신장질환(chronic kidney disease)의 정의를 기술하시오.",
}

with open("main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "QUESTIONS = [" not in src:
        continue
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
        print(f"[tuple {i}/{m.group(2)}] DB idx≈{i+1}")
        print(f"  신: {new_q}")
    cell["source"] = [new_src]
    break

with open("main.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ main.ipynb 패치 완료")
