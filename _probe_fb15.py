import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

CANDIDATES = [
    (3, "limiting sodium intake hypertension blood pressure"),
    (3, "limiting alcohol intake high blood pressure"),
    (9, "HbA1c test purpose diagnosis diabetes"),
    (9, "type 2 diabetes mellitus definition"),
    (13, "coronary artery disease definition atherosclerosis"),
    (13, "prevention coronary artery disease modifiable risk factors"),
    (31, "Alzheimer disease forgetting recent events early symptom"),
    (31, "Alzheimer disease progressive loss mental function definition"),
    (37, "SSRI selective serotonin reuptake inhibitors depression treatment"),
    (37, "major depressive disorder unipolar definition"),
    (38, "major depressive disorder difficulty falling asleep symptom"),
    (38, "major depressive disorder depressed mood characteristic symptom"),
    (43, "asthma definition reversible airway obstruction"),
    (43, "asthma pathophysiology bronchoconstriction"),
    (57, "lower esophageal sphincter prevents reflux esophagus"),
    (57, "GERD heartburn most prominent symptom"),
    (74, "iron deficiency anemia tired fatigue consumer"),
    (74, "iron deficiency anemia cause low iron"),
    (94, "colorectal cancer screening purpose detect cancer early"),
    (94, "colorectal cancer vegetables diet fiber risk reduction"),
    (95, "colorectal cancer screening colonoscopy purpose"),
    (96, "colorectal cancer screening colonoscopy detect polyps cancer"),
    (101, "peptic ulcer disease most common causes H pylori NSAID"),
    (102, "peptic ulcer disease definition erosion stomach duodenum"),
    (103, "peptic ulcer disease stomach acid digestive juices"),
]

for idx, q in CANDIDATES:
    r = search_msd_manual.invoke({"query": q})
    ok = len(r) > 400
    print(f"idx~{idx} ok={ok} len={len(r)} | {q[:55]}")
    print(f"  {r[250:520].replace(chr(10), ' ')}")
    print()
