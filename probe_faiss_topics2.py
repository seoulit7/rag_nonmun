import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

queries = [
    ("HbA1c", "HbA1c diagnostic criteria type 2 diabetes"),
    ("CURB", "CURB-65 pneumonia severity score"),
    ("rt-PA", "thrombolytic therapy rt-PA stroke"),
    ("DSM", "DSM-5 major depressive disorder criteria"),
    ("MDD symptoms", "major depressive disorder characteristic symptoms signs"),
    ("stroke risk ko", "modifiable risk factors stroke hypertension"),
    ("diabetes symptom", "type 2 diabetes overweight insulin resistance symptoms"),
    ("thyroid cause", "primary hypothyroidism causes Hashimoto"),
    ("pneumonia symptom only", "pneumonia cough fever sputum symptoms"),
]

for label, q in queries:
    r = search_msd_manual.invoke({"query": q})
    hit = label.lower() in r.lower() or q.split()[0].lower() in r.lower()
    print(f"=== {label} | hit={hit} | len={len(r)} ===")
    for kw in ["HbA1c", "CURB", "rt-PA", "thromboly", "DSM", "depressed mood", "anhedonia", "Hashimoto", "cough", "risk factors"]:
        if kw.lower() in r.lower():
            print(f"  found: {kw}")
    print(r[:500].replace("\n", " ")[:500])
    print()
