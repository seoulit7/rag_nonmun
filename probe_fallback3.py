import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

queries = [
    ("COPD smoking P", "COPD smoking causes chronic inflammation airways"),
    ("COPD GOLD", "COPD GOLD classification FEV1 severity"),
    ("depression symptoms C", "major depressive disorder characteristic symptoms signs"),
    ("depression vs sadness", "differentiate depression from normal sadness"),
    ("anemia iron P", "iron deficiency anemia serum ferritin diagnosis"),
    ("anemia causes P", "iron deficiency anemia causes classification"),
]

for label, q in queries:
    r = search_msd_manual.invoke({"query": q})
    ok = len(r) > 200 and "not contain" not in r[:400].lower()
    print(f"{label}: ok={ok} len={len(r)}")
    print(r[200:450].replace(chr(10), " ")[:200])
    print()
