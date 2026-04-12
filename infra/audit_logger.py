"""Supabase rag_audit_log 테이블에 RAGAS 평가 결과를 저장·업데이트하는 모듈."""
import logging
import threading
from typing import Optional

import psycopg2
import psycopg2.extras

import config.settings as settings
from models.state import GraphState

logger = logging.getLogger(__name__)

# 커넥션은 스레드별로 관리 (Streamlit 멀티스레드 환경 대응)
_local = threading.local()


def _get_conn():
    """스레드-로컬 psycopg2 커넥션을 반환한다. 끊어졌으면 재연결."""
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed:
        url = settings.SUPABASE_DB_URL
        if not url:
            raise RuntimeError("SUPABASE_DB_URL이 설정되지 않았습니다.")
        conn = psycopg2.connect(url)
        conn.autocommit = True
        _local.conn = conn
    return conn


# ──────────────────────────────────────────────────────────────────────────────
# INSERT: critic 평가 직후 호출
# ──────────────────────────────────────────────────────────────────────────────

def save_audit_log(
    state: GraphState,
    request_id: str,
    is_escalated: bool,
    is_fallback: bool,
    execution_time_ms: Optional[int] = None,
) -> Optional[int]:
    """RAGAS 평가 결과를 rag_audit_log 테이블에 INSERT하고 log_id를 반환한다.

    호출 시점: _critic_node() 내부에서 라우팅 분기 결정 직후 (output_agent 실행 전).
    final_answer는 아직 번역되지 않아 NULL로 저장하고, output/fallback 후
    update_audit_log_answer()로 업데이트한다.

    Args:
        state:             현재 LangGraph GraphState
        request_id:        워크플로우 전체 고유 ID (UUID 문자열)
        is_escalated:      이번 평가에서 상위 Tier로 에스컬레이션 됐는지 여부
        is_fallback:       모든 Tier 소진으로 fallback 노드로 라우팅됐는지 여부
        execution_time_ms: critic_agent() (RAGAS 평가) 소요 시간 (밀리초)

    Returns:
        삽입된 log_id (int), 실패 시 None
    """
    user_level = state.get("user_level") or ""
    # DB CHECK: user_level IN ('Professional', 'Consumer') — 분류 전이면 저장 스킵
    if user_level not in ("Professional", "Consumer"):
        logger.warning("[AuditLog] user_level=%r 이 유효하지 않아 저장을 건너뜁니다.", user_level)
        return None

    optimized_query: Optional[str] = (
        state["queries"][-1] if state.get("queries") else None
    )

    row = {
        "request_id":        request_id,
        "user_level":        user_level,
        "original_query":    state["question"],
        "optimized_query":   optimized_query,
        "final_answer":      None,                          # output 후 UPDATE
        "tier_id":           state["search_tier"],
        "loop_count":        state["loop_count"],
        "ragas_f":           float(state.get("critic_score", 0.0)),
        "ragas_ar":          float(state.get("answer_relevance_score", 0.0)),
        "ragas_cp":          float(state.get("context_precision_score", 0.0)),
        "is_escalated":      is_escalated,
        "is_fallback":       is_fallback,
        "retrieved_doc_count": len(state.get("context") or []),
        "llm_model":         state.get("llm_provider") or "openai",
        "execution_time_ms": execution_time_ms,
    }

    sql = """
        INSERT INTO public.rag_audit_log (
            request_id, user_level, original_query, optimized_query, final_answer,
            tier_id, loop_count,
            ragas_f, ragas_ar, ragas_cp,
            is_escalated, is_fallback,
            retrieved_doc_count, llm_model, execution_time_ms
        ) VALUES (
            %(request_id)s, %(user_level)s, %(original_query)s, %(optimized_query)s, %(final_answer)s,
            %(tier_id)s, %(loop_count)s,
            %(ragas_f)s, %(ragas_ar)s, %(ragas_cp)s,
            %(is_escalated)s, %(is_fallback)s,
            %(retrieved_doc_count)s, %(llm_model)s, %(execution_time_ms)s
        )
        RETURNING log_id;
    """

    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, row)
            log_id: int = cur.fetchone()[0]
        logger.info(
            "[AuditLog] INSERT log_id=%d | request_id=%s | tier=%d loop=%d "
            "F=%.3f AR=%.3f CP=%.3f escalated=%s fallback=%s",
            log_id, request_id,
            row["tier_id"], row["loop_count"],
            row["ragas_f"], row["ragas_ar"], row["ragas_cp"],
            is_escalated, is_fallback,
        )
        return log_id
    except Exception as exc:
        logger.error("[AuditLog] INSERT 실패: %s", exc, exc_info=True)
        # 커넥션이 오염됐을 수 있으므로 초기화
        _local.conn = None
        return None


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE: output_agent / fallback_node 실행 후 호출
# ──────────────────────────────────────────────────────────────────────────────

def update_audit_log_answer(request_id: str, final_answer: str) -> None:
    """같은 request_id의 모든 행에 번역 완료된 최종 답변을 UPDATE한다.

    호출 시점: output_agent 또는 fallback_node 완료 직후.
    한 request_id 내 여러 루프 행이 있을 수 있으나, final_answer는 마지막 답변만
    의미 있으므로 request_id 전체에 일괄 UPDATE한다.

    Args:
        request_id:   워크플로우 전체 고유 ID (UUID 문자열)
        final_answer: output_agent가 생성한 한국어 최종 답변
    """
    sql = """
        UPDATE public.rag_audit_log
        SET final_answer = %(answer)s
        WHERE request_id = %(request_id)s::uuid;
    """
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, {"answer": final_answer, "request_id": request_id})
        logger.info("[AuditLog] UPDATE final_answer | request_id=%s", request_id)
    except Exception as exc:
        logger.error("[AuditLog] UPDATE 실패: %s", exc, exc_info=True)
        _local.conn = None
