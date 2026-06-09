import sys; sys.stdout.reconfigure(encoding='utf-8')
keys = ["OPENAI_MODEL", "MEDICAL_RAG_RAGAS_LLM_MODEL", "GEMINI_MODEL", "MEDICAL_RAG_GEMINI_AUX_MODEL",
        "MEDICAL_RAG_CLASSIFIER_MODEL", "MEDICAL_RAG_TRANSLATE_MODEL"]
with open('.env', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        k = line.split('=', 1)[0].strip()
        if k in keys:
            print(line)
