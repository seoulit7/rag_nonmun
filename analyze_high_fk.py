import sys; sys.stdout.reconfigure(encoding='utf-8')
import re
import psycopg2
import config.settings as settings
from infra.evaluator import get_pure_fk_grade, flesch_kincaid_grade_en, _PURE_FK_EXPLICIT_PAT, _PURE_FK_SUFFIX, _PURE_FK_EPONYM, _PURE_FK_LONG

conn = psycopg2.connect(settings.SUPABASE_DB_URL)
cur = conn.cursor()

# Consumer FK >= 10 인 최신 is_final 행 (final_answer 포함)
cur.execute("""
    SELECT query_index, disease, original_query, fk_grade, final_answer
    FROM public.rag_audit_log
    WHERE is_final = TRUE AND user_level = 'Consumer' AND fk_grade >= 10
    ORDER BY fk_grade DESC
""")
rows = cur.fetchall()
conn.close()

for qi, dis, q, fk, ans in rows:
    print("=" * 70)
    print(f"idx={qi} {dis} FK={fk}")
    print(f"Q: {q[:80]}")
    if not ans:
        print("  (final_answer 없음 — fallback 원문)")
        continue
    # 영어 부분만 추출 (번역 전 답변이 final_answer에 포함되어 있는지 확인)
    # final_answer는 한국어 번역본 — FK는 번역 전 영어 원문으로 계산됨
    # 단어 분석: 마스킹 후 남은 고음절 단어 식별
    # final_answer에서 영어 단어만 추출
    english_words = re.findall(r'\b[a-zA-Z]+\b', ans)
    print(f"영어단어 {len(english_words)}개")
    print(f"  (답변 앞부분 500자): {ans[:500]}")
    print()
    # 마스킹 시뮬레이션 (영어 원문이 있다면)
    # final_answer는 한국어지만 영어 의료 용어가 섞여 있을 수 있음
    if len(english_words) > 5:
        masked = _PURE_FK_SUFFIX.sub("it", ans)
        masked = _PURE_FK_EPONYM.sub("it", masked)
        masked = _PURE_FK_LONG.sub("it", masked)
        masked = _PURE_FK_EXPLICIT_PAT.sub("it", masked)
        fk_sim = flesch_kincaid_grade_en(masked)
        print(f"  마스킹 후 FK 시뮬 (한국어+영어 혼재): {fk_sim:.2f}")
        # 마스킹 안 된 영어 단어
        remaining_en = re.findall(r'\b[a-zA-Z]{4,}\b', masked)
        from collections import Counter
        cnt = Counter(remaining_en)
        print(f"  남은 영어단어 상위 20: {cnt.most_common(20)}")
