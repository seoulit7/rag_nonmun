import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

CANDIDATES = [
    (37, "major depressive disorder definition unipolar"),
    (37, "major depressive disorder sleep disturbance symptom"),
    (37, "major depressive disorder weight change symptom"),
    (38, "major depressive disorder depressed mood symptom"),
    (38, "major depressive disorder psychomotor agitation symptom"),
    (57, "gastroesophageal reflux disease GERD definition"),
    (57, "GERD heartburn most prominent symptom cause"),
    (57, "lower esophageal sphincter function GERD"),
    (93, "colorectal cancer rectal bleeding symptom"),
    (93, "colorectal cancer screening colonoscopy purpose"),
    (93, "colorectal cancer blood in stool evaluation"),
    (101, "peptic ulcer disease definition"),
    (101, "Helicobacter pylori peptic ulcer cause"),
    (101, "NSAID peptic ulcer mechanism prostaglandin"),
]

for idx, q in CANDIDATES:
    r = search_msd_manual.invoke({"query": q})
    ok = len(r) > 300
    print(f"idx~{idx} ok={ok} len={len(r)}")
    print(f"  Q: {q}")
    print(f"  SNIP: {r[200:450].replace(chr(10), ' ')}")
    print()
