"""
Condition A에서 Tier 1/2로 에스컬레이션된 질문 조회.

사용법:
    python check_tier_escalation.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import oracledb
import config.settings as settings
from stqs_questions import QUESTIONS

# query_index → question text 맵
Q_MAP = {q[2]: (q[0], q[1], q[3]) for q in QUESTIONS}  # {idx: (disease, level, text)}


def main():
    conn = oracledb.connect(
        user=settings.ORACLE_USER,
        password=settings.ORACLE_PASSWORD,
        dsn=settings.ORACLE_DSN,
    )

    with conn.cursor() as cur:
        # Condition A 최종 행만, 티어별 집계
        cur.execute("""
            SELECT query_index, final_tier, query_level_label, disease
            FROM rag_audit_log
            WHERE ablation_condition = 'A'
              AND is_final = 1
              AND query_index IS NOT NULL
            ORDER BY query_index
        """)
        rows = cur.fetchall()
    conn.close()

    # 집계
    tier_map = {}  # query_index → final_tier
    for query_index, final_tier, level, disease in rows:
        tier_map[int(query_index)] = (int(final_tier), level, disease)

    tier0 = [(idx, v) for idx, v in tier_map.items() if v[0] == 0]
    tier1 = [(idx, v) for idx, v in tier_map.items() if v[0] == 1]
    tier2 = [(idx, v) for idx, v in tier_map.items() if v[0] == 2]

    print("=" * 70)
    print(f"  Condition A 에스컬레이션 현황  (전체 {len(tier_map)}건)")
    print(f"  Tier 0: {len(tier0)}건  |  Tier 1: {len(tier1)}건  |  Tier 2: {len(tier2)}건")
    print("=" * 70)

    if tier1:
        print(f"\n▶ Tier 1 에스컬레이션 ({len(tier1)}건)")
        print("-" * 70)
        for idx, (tier, level, disease) in sorted(tier1):
            text = Q_MAP.get(idx, ("?", "?", "?"))[2]
            print(f"  [{idx:>3}] [{level}] {disease}")
            print(f"        {text[:90]}{'...' if len(text) > 90 else ''}")

    if tier2:
        print(f"\n▶ Tier 2 에스컬레이션 ({len(tier2)}건)")
        print("-" * 70)
        for idx, (tier, level, disease) in sorted(tier2):
            text = Q_MAP.get(idx, ("?", "?", "?"))[2]
            print(f"  [{idx:>3}] [{level}] {disease}")
            print(f"        {text[:90]}{'...' if len(text) > 90 else ''}")

    print()
    target = (len(tier1) + len(tier2)) // 2
    print(f"  ▶ 튜닝 목표: Tier 1+2 ({len(tier1)+len(tier2)}건) 중 {target}건을 Tier 0으로 이동")
    print("=" * 70)


if __name__ == "__main__":
    main()
