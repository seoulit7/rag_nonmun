"""Oracle rag_audit_log 테이블 전체 삭제 스크립트.

실험 질문이 영어로 교체되었으므로 기존 한글 질문 기반 데이터를 전부 삭제하고
새로 480회(조건 A·E × 240문항) 재실행한다.

사용법:
    python reset_audit_log.py           # 실제 삭제 (확인 프롬프트 있음)
    python reset_audit_log.py --dry-run # 삭제 건수만 확인, 실제 삭제 안 함
"""
import sys

import oracledb

import config.settings as settings

DRY_RUN = "--dry-run" in sys.argv


def main() -> None:
    conn = oracledb.connect(
        user=settings.ORACLE_USER,
        password=settings.ORACLE_PASSWORD,
        dsn=settings.ORACLE_DSN,
    )

    with conn.cursor() as cur:
        # ── 삭제 대상 건수 확인 ──────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) FROM rag_audit_log")
        total: int = cur.fetchone()[0]

        cur.execute(
            "SELECT ablation_condition, COUNT(*) n "
            "FROM rag_audit_log "
            "GROUP BY ablation_condition "
            "ORDER BY ablation_condition"
        )
        by_cond = cur.fetchall()

    print("=" * 60)
    print("  rag_audit_log 삭제 대상 현황")
    print("=" * 60)
    print(f"  전체 행 수: {total:,}")
    if by_cond:
        print("  조건별 분포:")
        for cond, cnt in by_cond:
            print(f"    [{cond or 'NULL'}]  {cnt:,}건")
    print("=" * 60)

    if total == 0:
        print("  삭제할 데이터가 없습니다.")
        conn.close()
        return

    if DRY_RUN:
        print("  [DRY-RUN] 위 데이터가 삭제 대상입니다. 실제 삭제는 수행하지 않습니다.")
        conn.close()
        return

    # ── 확인 프롬프트 ────────────────────────────────────────────────────────
    answer = input(f"\n  위 {total:,}건을 전부 삭제하시겠습니까? (yes 입력 시 실행): ").strip()
    if answer.lower() != "yes":
        print("  취소되었습니다.")
        conn.close()
        return

    # ── 실제 삭제 ────────────────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rag_audit_log")
        deleted = cur.rowcount
    conn.commit()
    conn.close()

    print(f"\n  완료: {deleted:,}건 삭제됨.")
    print("  이제 main.ipynb 에서 영어 질문으로 480회 재실행하세요.")


if __name__ == "__main__":
    main()
