import sys

sys.stdout.reconfigure(encoding="utf-8")
from infra.evaluator import get_pure_fk_grade

text_hyp = (
    "Hypertension increases stroke risk by narrowing arteries and promoting arterial deposits. "
    "Elevated blood pressure damages arterial walls over time. "
    "This narrowing reduces blood flow to the brain. "
    "Damaged arteries can rupture, causing bleeding into brain tissue. "
    "Blood clots can also form in narrowed arteries and block cerebral blood flow."
)
text_anemia = (
    "Iron-deficiency anemia causes fatigue and shortness of breath because the body lacks hemoglobin. "
    "Hemoglobin is the protein in erythrocytes that carries oxygen to tissues. "
    "Without sufficient iron, the bone marrow produces fewer red blood cells. "
    "Reduced oxygen carrying capacity leads to fatigue and decreased endurance. "
    "Serum ferritin levels below normal confirm iron deficiency anemia."
)
text_anxiety = (
    "Anxiety disorder triggers the autonomic nervous system, activates the fight-or-flight response. "
    "This cascade of signals releases adrenaline. "
    "Adrenaline causes the heart to beat faster and triggers sweating. "
    "The body mediates these physical symptoms via stress pathways. "
    "These pathogens of anxiety cause palpitations and sweating."
)

print(f"고혈압 시뮬 FK: {get_pure_fk_grade(text_hyp):.2f}  (목표 ≤ 9)")
print(f"빈혈 시뮬 FK:   {get_pure_fk_grade(text_anemia):.2f}  (목표 ≤ 9)")
print(f"불안장애 시뮬 FK: {get_pure_fk_grade(text_anxiety):.2f}  (목표 ≤ 9)")
