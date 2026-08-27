import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from langchain_community.document_loaders import PyPDFLoader

missing = [
    ("강직성척추염",
     "Ankylosing Spondylitis - Rheumatology and Orthopedics - MSD Manual Professional Edition.pdf",
     "Ankylosing Spondylitis - Bone, Joint, and Muscle Disorders - MSD Manual Consumer Version.pdf"),
    ("재생불량성빈혈",
     "Aplastic Anemia - Hematology - MSD Manual Professional Edition.pdf",
     "Aplastic Anemia - Blood Disorders - MSD Manual Consumer Version.pdf"),
    ("충수염",
     "Appendicitis - Gastroenterology - MSD Manual Professional Edition.pdf",
     "Appendicitis - Digestive Disorders - MSD Manual Consumer Version.pdf"),
    ("COVID-19",
     "COVID-19 - Infectious Disease - MSD Manual Professional Edition.pdf",
     "COVID-19 - Infections - MSD Manual Consumer Version.pdf"),
    ("담석증",
     "Cholelithiasis - Hepatology - MSD Manual Professional Edition.pdf",
     "Gallstones - Liver and Gallbladder Disorders - MSD Manual Consumer Version.pdf"),
    ("만성췌장염",
     "Chronic Pancreatitis - Gastroenterology - MSD Manual Professional Edition.pdf",
     "Chronic Pancreatitis - Digestive Disorders - MSD Manual Consumer Version.pdf"),
    ("자궁내막증",
     "Endometriosis - Gynecology and Obstetrics - MSD Manual Professional Edition.pdf",
     "Endometriosis - Women's Health - MSD Manual Consumer Version.pdf"),
    ("HIV감염",
     "Human Immunodeficiency Virus (HIV) Infection - Infectious Disease - MSD Manual Professional Edition.pdf",
     "Human Immunodeficiency Virus (HIV) Infection - Infections - MSD Manual Consumer Version.pdf"),
    ("과민성대장증후군",
     "Irritable Bowel Syndrome (IBS) - Gastroenterology - MSD Manual Professional Edition.pdf",
     "Irritable Bowel Syndrome (IBS) - Digestive Disorders - MSD Manual Consumer Version.pdf"),
    ("요추척추관협착증",
     "Lumbar Spinal Stenosis - Rheumatology and Orthopedics - MSD Manual Professional Edition.pdf",
     "Lumbar Spinal Stenosis - Bone, Joint, and Muscle Disorders - MSD Manual Consumer Version.pdf"),
    ("폐암",
     "Lung Carcinoma - Oncology - MSD Manual Professional Edition.pdf",
     "Lung Cancer - Cancer - MSD Manual Consumer Version.pdf"),
    ("편두통",
     "Migraine - Neurology - MSD Manual Professional Edition.pdf",
     "Migraines - Brain, Spinal Cord, and Nerve Disorders - MSD Manual Consumer Version.pdf"),
    ("골반염",
     "Pelvic Inflammatory Disease (PID) - Gynecology and Obstetrics - MSD Manual Professional Edition.pdf",
     "Pelvic Inflammatory Disease (PID) - Women's Health - MSD Manual Consumer Version.pdf"),
    ("전립선암",
     "Prostate Cancer - Oncology - MSD Manual Professional Edition.pdf",
     "Prostate Cancer - Cancer - MSD Manual Consumer Version.pdf"),
    ("건선",
     "Psoriasis - Dermatology - MSD Manual Professional Edition.pdf",
     "Psoriasis - Skin Disorders - MSD Manual Consumer Version.pdf"),
    ("소아청소년제2형당뇨병",
     "Type 2 Diabetes Mellitus in Children and Adolescents - Pediatrics - MSD Manual Professional Edition.pdf",
     "Type 2 Diabetes in Children and Adolescents - Children's Health - MSD Manual Consumer Version.pdf"),
    ("두드러기",
     "Urticaria - Dermatology - MSD Manual Professional Edition.pdf",
     "Hives - Skin Disorders - MSD Manual Consumer Version.pdf"),
    ("자궁근종",
     "Uterine Fibroids - Gynecology and Obstetrics - MSD Manual Professional Edition.pdf",
     "Uterine Fibroids - Women's Health - MSD Manual Consumer Version.pdf"),
]

for name, pro_file, con_file in missing:
    for label, fname in [("PRO", pro_file), ("CON", con_file)]:
        try:
            loader = PyPDFLoader(f"data/{fname}")
            pages = loader.load()
            text = " ".join(p.page_content for p in pages)[:1800]
            print(f"=={name}|{label}==")
            print(text)
            print()
        except Exception as e:
            print(f"=={name}|{label}== ERROR: {e}")
            print()
