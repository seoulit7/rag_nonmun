import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

queries = [
    ("diabetes P", "type 2 diabetes symptoms hyperglycemia"),
    ("diabetes alt", "metformin mechanism type 2 diabetes"),
    ("diabetes alt2", "diagnosis type 2 diabetes fasting glucose"),
    ("stroke P", "stroke risk factors hypertension diabetes"),
    ("stroke alt", "transient ischemic attack definition"),
    ("stroke alt2", "ischemic stroke symptoms facial weakness"),
    ("depression P", "major depressive disorder symptoms diagnosis"),
    ("depression alt", "selective serotonin reuptake inhibitors SSRIs"),
    ("thyroid P", "primary hypothyroidism Hashimoto thyroiditis cause"),
    ("thyroid alt", "hypothyroidism symptoms fatigue cold intolerance"),
    ("CAP P", "community acquired pneumonia clinical presentation"),
    ("CAP alt", "pneumonia antibiotic treatment outpatient"),
    ("CAP C", "pneumonia symptoms fever cough"),
]

for label, q in queries:
    r = search_msd_manual.invoke({"query": q})
    ok = len(r) > 150 and "not contain" not in r.lower()[:300]
    print(f"--- {label} | ok={ok} | len={len(r)} ---")
    print(r[:350].replace("\n", " "))
    print()
