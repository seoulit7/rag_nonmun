"""rag_audit_log 테이블 조회 함수 모음."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

import config.settings as settings

logger = logging.getLogger(__name__)

PAGE_SIZE = 20

TIER_LABEL = {0: "Tier 0 · VectorDB", 1: "Tier 1 · LLM", 2: "Tier 2 · Web"}


def _conn():
    return psycopg2.connect(settings.SUPABASE_DB_URL)


# ── 목록 조회 ─────────────────────────────────────────────────────────────────

def fetch_logs(
    date_from: date | None = None,
    date_to: date | None = None,
    user_levels: list[str] | None = None,
    tiers: list[int] | None = None,
    escalated: bool | None = None,
    fallback: bool | None = None,
    ragas_f_min: float = 0.0,
    ragas_f_max: float = 1.0,
    keyword: str = "",
    page: int = 1,
) -> tuple[pd.DataFrame, int]:
    """필터 조건으로 rag_audit_log를 페이지네이션 조회한다.

    Returns:
        (DataFrame, 전체 건수)
    """
    where: list[str] = []
    params: list[Any] = []

    if date_from:
        where.append("created_at >= %s")
        params.append(datetime(date_from.year, date_from.month, date_from.day,
                               tzinfo=timezone.utc))
    if date_to:
        where.append("created_at < %s")
        # date_to 당일 포함이므로 +1일
        from datetime import timedelta
        next_day = date_to + timedelta(days=1)
        params.append(datetime(next_day.year, next_day.month, next_day.day,
                               tzinfo=timezone.utc))
    if user_levels:
        where.append("user_level = ANY(%s)")
        params.append(user_levels)
    if tiers:
        where.append("tier_id = ANY(%s)")
        params.append(tiers)
    if escalated is not None:
        where.append("is_escalated = %s")
        params.append(escalated)
    if fallback is not None:
        where.append("is_fallback = %s")
        params.append(fallback)

    where.append("ragas_f BETWEEN %s AND %s")
    params.extend([ragas_f_min, ragas_f_max])

    if keyword.strip():
        where.append("original_query ILIKE %s")
        params.append(f"%{keyword.strip()}%")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    offset = (page - 1) * PAGE_SIZE

    count_sql = f"SELECT COUNT(*) FROM public.rag_audit_log {where_sql}"
    data_sql = f"""
        SELECT
            log_id,
            request_id,
            created_at AT TIME ZONE 'Asia/Seoul' AS created_at,
            user_level,
            original_query,
            tier_id,
            loop_count,
            ragas_f,
            ragas_ar,
            ragas_cp,
            is_escalated,
            is_fallback,
            execution_time_ms
        FROM public.rag_audit_log
        {where_sql}
        ORDER BY created_at DESC
        LIMIT {PAGE_SIZE} OFFSET {offset}
    """

    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(count_sql, params)
                total: int = cur.fetchone()[0]

                cur.execute(data_sql, params)
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(), total

        df = pd.DataFrame([dict(r) for r in rows])
        df["request_id_short"] = df["request_id"].astype(str).str[:8] + "..."
        df["tier_label"] = df["tier_id"].map(TIER_LABEL)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        return df, total

    except Exception as e:
        logger.error("fetch_logs 실패: %s", e, exc_info=True)
        return pd.DataFrame(), 0


# ── 상세 조회 ─────────────────────────────────────────────────────────────────

def fetch_detail(request_id: str) -> dict:
    """단일 request_id의 전체 루프 이력과 메타 정보를 반환한다.

    Returns:
        {
            "meta": {...},          # 첫 번째 행의 요청 정보
            "loops": DataFrame,     # 전체 루프 행 (tier_id, loop_count 순)
            "queries": [...],       # optimized_query 이력 (중복 제거 순서 유지)
            "final_answer": str,    # 마지막 행의 final_answer
        }
    """
    sql = """
        SELECT
            log_id,
            request_id,
            created_at AT TIME ZONE 'Asia/Seoul' AS created_at,
            user_level,
            original_query,
            optimized_query,
            final_answer,
            tier_id,
            loop_count,
            ragas_f,
            ragas_ar,
            ragas_cp,
            is_escalated,
            is_fallback,
            retrieved_doc_count,
            llm_model,
            execution_time_ms
        FROM public.rag_audit_log
        WHERE request_id = %s::uuid
        ORDER BY tier_id ASC, loop_count ASC
    """
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(sql, (request_id,))
                rows = cur.fetchall()

        if not rows:
            return {}

        dicts = [dict(r) for r in rows]
        first = dicts[0]

        loops_df = pd.DataFrame(dicts)
        loops_df["tier_label"] = loops_df["tier_id"].map(TIER_LABEL)
        loops_df["created_at"] = pd.to_datetime(loops_df["created_at"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # optimized_query 이력 (순서 유지, 중복 제거)
        seen: set[str] = set()
        queries: list[str] = []
        for d in dicts:
            q = (d.get("optimized_query") or "").strip()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)

        final_answer = dicts[-1].get("final_answer") or ""

        return {
            "meta": {
                "request_id":    str(first["request_id"]),
                "created_at":    pd.to_datetime(first["created_at"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "user_level":    first["user_level"],
                "llm_model":     first["llm_model"] or "",
                "original_query": first["original_query"],
            },
            "loops":        loops_df,
            "queries":      queries,
            "final_answer": final_answer,
        }

    except Exception as e:
        logger.error("fetch_detail 실패: %s", e, exc_info=True)
        return {}
