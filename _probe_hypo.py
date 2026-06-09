import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

for q in [
    "hypoglycemia treatment glucose tablets juice",
    "low blood sugar diabetes fast acting carbohydrate",
    "diabetes hypoglycemia symptoms treatment",
]:
    r = search_msd_manual.invoke({"query": q})
    print("===", q)
    print(r[:500])
    print()
