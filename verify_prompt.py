import ast, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('agents/rag_engine.py', encoding='utf-8') as f:
    src = f.read()
count = src.count('Begin with ONE direct summary sentence')
print(f'summary sentence 규칙 삽입 횟수: {count}곳 (기대값: 3)')
try:
    ast.parse(src)
    print('구문 오류 없음')
except SyntaxError as e:
    print(f'구문 오류: {e}')
