import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

CANDIDATES = [
    (3, "고혈압", "P", "DASH diet hypertension blood pressure reduction"),
    (3, "고혈압", "P", "weight loss hypertension blood pressure"),
    (3, "고혈압", "P", "hypertension lifestyle modification exercise"),
    (13, "관상동맥", "P", "coronary artery disease atherosclerosis definition"),
    (13, "관상동맥", "P", "acute coronary syndrome myocardial infarction"),
    (13, "관상동맥", "P", "coronary artery disease risk factors smoking"),
    (25, "COPD", "P", "chronic obstructive pulmonary disease definition"),
    (25, "COPD", "P", "COPD smoking causes chronic bronchitis emphysema"),
    (26, "COPD", "P", "COPD smoking risk factor pathogenesis"),
    (26, "COPD", "P", "COPD chronic bronchitis emphysema definition"),
    (31, "알츠하이머", "P", "Alzheimer disease definition dementia"),
    (31, "알츠하이머", "P", "Alzheimer disease amyloid plaques neurofibrillary tangles"),
    (31, "알츠하이머", "P", "Alzheimer disease memory loss early symptoms"),
    (37, "우울증", "P", "major depressive disorder SSRI serotonin reuptake"),
    (37, "우울증", "P", "major depressive disorder first line treatment antidepressant"),
    (37, "우울증", "P", "major depressive disorder definition"),
    (38, "우울증", "P", "major depressive disorder diagnostic criteria symptoms"),
    (38, "우울증", "P", "depressed mood anhedonia major depression"),
    (41, "우울증", "C", "major depressive disorder symptoms sadness"),
    (41, "우울증", "C", "depression symptoms sleep appetite energy"),
    (44, "천식", "P", "asthma bronchoconstriction airway inflammation"),
    (44, "천식", "P", "asthma definition reversible airway obstruction"),
    (44, "천식", "P", "asthma bronchospasm smooth muscle"),
    (57, "GERD", "P", "gastroesophageal reflux disease lower esophageal sphincter"),
    (57, "GERD", "P", "GERD definition heartburn acid reflux"),
    (57, "GERD", "P", "GERD risk factors obesity hiatal hernia"),
    (61, "GERD", "C", "GERD heartburn acid reflux symptoms"),
    (61, "GERD", "C", "gastroesophageal reflux esophagitis complications"),
    (61, "GERD", "C", "GERD lifestyle avoid lying down after eating"),
    (93, "대장암", "C", "colorectal cancer screening colonoscopy age"),
    (93, "대장암", "C", "colorectal cancer rectal bleeding symptoms"),
    (93, "대장암", "C", "colorectal cancer risk factors diet"),
    (101, "위궤양", "P", "peptic ulcer disease definition"),
    (101, "위궤양", "P", "NSAID peptic ulcer gastritis mechanism"),
    (101, "위궤양", "P", "Helicobacter pylori peptic ulcer"),
]

for idx, dis, lvl, q in CANDIDATES:
    r = search_msd_manual.invoke({"query": q})
    ok = len(r) > 300 and "does not contain" not in r.lower()[:500]
    print(f"idx~{idx} {dis} [{lvl}] ok={ok} len={len(r)}")
    print(f"  Q: {q}")
    print(f"  SNIP: {r[:280].replace(chr(10), ' ')}")
    print()
