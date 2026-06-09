import sys

sys.stdout.reconfigure(encoding="utf-8")

from tools.vector_search import initialize_vector_db, search_msd_manual

initialize_vector_db()

topics = [
    (3, "nicotine vasoconstrictor hypertension blood pressure"),
    (12, "hypoglycemia 15 grams simple carbohydrate"),
    (14, "stable angina exertion myocardial ischemia"),
    (15, "myocardial infarction reperfusion restore blood flow"),
    (17, "heart attack symptoms call emergency immediately"),
    (19, "stroke definition focal neurological deficit"),
    (26, "COPD chronic bronchitis cigarette smoke"),
    (27, "COPD exacerbation respiratory infection"),
    (38, "major depressive disorder serotonin"),
    (39, "electroconvulsive therapy depression"),
    (48, "asthma short acting beta agonist albuterol"),
    (53, "ACE inhibitor chronic kidney disease"),
    (56, "sodium restriction chronic kidney disease edema"),
    (59, "Barrett esophagus gastroesophageal reflux"),
    (68, "levothyroxine empty stomach morning"),
    (73, "anemia hemoglobin fatigue shortness of breath"),
    (76, "generalized anxiety disorder excessive worry"),
    (80, "relaxation breathing anxiety symptoms"),
    (84, "community acquired pneumonia blood culture"),
    (91, "Lynch syndrome colorectal cancer screening"),
]

for idx, q in topics:
    r = search_msd_manual.invoke({"query": q})
    low = r.lower()
    ok = len(r) > 200 and "does not contain" not in low and "not contain sufficient" not in low
    print(f"[{idx:3d}] ok={ok} len={len(r)}")
    print(f"  Q: {q}")
    if ok:
        print(f"  -> {r[:220].replace(chr(10), ' ')}")
