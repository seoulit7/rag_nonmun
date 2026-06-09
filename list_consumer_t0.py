import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("main.ipynb", encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "QUESTIONS = [" in src and "assert len(QUESTIONS)" in src:
        g = {}
        exec(src, g)
        qs = g["QUESTIONS"]
        break

for i, (d, lv, tier, q) in enumerate(qs):
    if lv == "C" and tier == 0:
        print(f"tuple[{i:2d}] DB~{i+1} {d} T{tier}")
        print(f"  {q[:85]}")
