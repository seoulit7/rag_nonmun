import sys

sys.stdout.reconfigure(encoding="utf-8")

import stqs_questions as s

for i in range(72, 81):
    d, l, t, q = s.QUESTIONS[i - 1]
    print(f"idx={i} {d} {l} | {q[:60]}")
