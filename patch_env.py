import sys; sys.stdout.reconfigure(encoding='utf-8')

PATCHES = {
    "OPENAI_MODEL": "gpt-4.1",
    "MEDICAL_RAG_RAGAS_LLM_MODEL": "gpt-4o",
}

with open('.env', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.strip()
    if stripped and not stripped.startswith('#') and '=' in stripped:
        k = stripped.split('=', 1)[0].strip()
        if k in PATCHES:
            new_val = PATCHES[k]
            old_line = line.rstrip()
            new_line = f"{k}={new_val}\n"
            new_lines.append(new_line)
            print(f"  변경: {old_line}  →  {new_line.rstrip()}")
            continue
    new_lines.append(line)

with open('.env', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ .env 저장 완료")
