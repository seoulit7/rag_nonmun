import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'QUESTIONS = [' in src:
        print(f"=== cell {i} ===")
        print(src[:12000])
        break
