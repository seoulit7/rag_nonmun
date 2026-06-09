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
        for i, (d, lv, tier, q) in enumerate(g["QUESTIONS"]):
            if tier == 0:
                print(f"idx={i+1:2d} [{lv}] {d}")
                print(f"  {q}")
        break
