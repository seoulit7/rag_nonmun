"""
수정 대상 Consumer 질문 7개:
  idx 13: 우울증 C       — 2주제 → 단일 비교 질문
  idx 15: 천식 C         — 2주제 → 단일 대처 질문
  idx 19: 만성신장질환 C  — 야뇨증(MSD 미흡) → CKD 빈혈 기전 (fallback 해결)
  idx 23: 갑상선기능저하증C — 2주제 → 단일 기전 질문
  idx 25: 빈혈 C         — 2주제(증상+음식) → 단일 기전 (fallback 해결)
  idx 27: 불안장애 C      — 목록형 → 단일 기전 질문
  idx 35: 유방암 C       — 2주제 → 단일 행동 질문
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

PATCHES = {
    13: "우울증과 단순한 슬픔은 어떻게 구별할 수 있나요?",
    15: "천식 발작이 일어났을 때 어떻게 대처해야 하나요?",
    19: "만성 신장질환이 있으면 왜 빈혈이 생기나요?",
    23: "갑상선 호르몬이 부족하면 왜 신진대사가 느려지나요?",
    25: "철분 결핍성 빈혈이 있으면 왜 숨이 차고 피곤한가요?",
    27: "불안장애는 왜 가슴이 두근거리고 땀이 나는 신체 증상을 유발하나요?",
    35: "유방에 통증 없는 혹이 만져지면 어떻게 해야 하나요?",
}

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# QUESTIONS = [ 셀 찾기
target_cell = None
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'QUESTIONS = [' in src:
        target_cell = cell
        break

if target_cell is None:
    print("ERROR: QUESTIONS 셀을 찾지 못했습니다.")
    sys.exit(1)

src = ''.join(target_cell['source'])

# QUESTIONS 리스트를 파싱해 각 튜플의 마지막 요소(질문 문자열)만 교체
# 간단하게: 튜플 단위로 분리 후 idx 위치의 question 교체
import re

# 각 질문 항목을 찾아 교체 (4번째 줄 = question 문자열)
# 패턴: 4-튜플의 마지막 문자열 리터럴
tuples_found = list(re.finditer(
    r'\(\s*"([^"]+)",\s*\n\s*"([PCpc])",\s*\n\s*(\d),\s*\n\s*"((?:[^"\\]|\\.)*)",\s*\n\s*\)',
    src,
    re.MULTILINE
))

print(f"튜플 {len(tuples_found)}개 발견")

new_src = src
offset = 0  # 교체 후 위치 보정

for i, m in enumerate(tuples_found):
    old_q = m.group(4)
    if i in PATCHES:
        new_q = PATCHES[i]
        # 기존 질문 문자열(따옴표 포함) 위치
        q_start = m.start(4) + offset
        q_end   = m.end(4)   + offset
        new_src = new_src[:q_start] + new_q + new_src[q_end:]
        offset += len(new_q) - len(old_q)
        print(f"  [idx {i:2d}] 교체: {old_q[:50]!r}")
        print(f"         → {new_q!r}")

target_cell['source'] = [new_src]

with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ main.ipynb 저장 완료")
