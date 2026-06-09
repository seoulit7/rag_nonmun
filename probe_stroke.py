import sys
sys.stdout.reconfigure(encoding="utf-8")
from tools.vector_search import initialize_vector_db, search_msd_manual
initialize_vector_db()
for label, q in [
    ("stroke symptoms", "initial symptoms of stroke"),
    ("stroke def", "definition of ischemic stroke"),
    ("TIA", "transient ischemic attack definition"),
    ("stroke prev", "prevention of stroke modifiable risk factors"),
]:
    r = search_msd_manual.invoke({"query": q})
    print(label, "len", len(r), r[300:500].replace("\n", " ")[:180])
    print()
