"""Supabase rag_audit_log 테이블에 RAGAS 평가 결과를 저장하는 모듈.

설계 원칙 (v3.0):
- save_loop_log : critic 평가 완료마다 무조건 INSERT (is_final=False). loop_number=eval_count.
- save_audit_log: output_node / fallback_node 완료 후 1회 INSERT (is_final=True). final_answer 포함.
- request_id당 N+1행 (N=critic 평가 횟수). 루프별 점수 변화 추적 가능.
- UPDATE 없음. INSERT only.
"""
import logging
import threading
import time
from typing import Optional

import psycopg2

import config.settings as settings
from models.state import GraphState

logger = logging.getLogger(__name__)

_local = threading.local()

_INSERT_SQL = """
    INSERT INTO public.rag_audit_log (
        request_id, loop_number, is_final,
        ablation_condition, query_index, disease,
        query_level_label, user_level,
        original_query, optimized_query,
        expected_tier, final_tier, tier_path,
        is_escalated, is_fallback, self_correction_count,
        ragas_f, ragas_ar, ragas_cp, q_total,
        hallucination_detected, hallucination_count,
        retrieved_doc_count, llm_model, execution_time_ms,
        final_answer, fk_grade
    ) VALUES (
        %(request_id)s, %(loop_number)s, %(is_final)s,
        %(ablation_condition)s, %(query_index)s, %(disease)s,
        %(query_level_label)s, %(user_level)s,
        %(original_query)s, %(optimized_query)s,
        %(expected_tier)s, %(final_tier)s, %(tier_path)s,
        %(is_escalated)s, %(is_fallback)s, %(self_correction_count)s,
        %(ragas_f)s, %(ragas_ar)s, %(ragas_cp)s, %(q_total)s,
        %(hallucination_detected)s, %(hallucination_count)s,
        %(retrieved_doc_count)s, %(llm_model)s, %(execution_time_ms)s,
        %(final_answer)s, %(fk_grade)s
    )
    RETURNING log_id;
"""


def _get_conn():
    """스레드-로컬 psycopg2 커넥션 반환. 끊어졌으면 재연결."""
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed:
        url = settings.SUPABASE_DB_URL
        if not url:
            raise RuntimeError("SUPABASE_DB_URL이 설정되지 않았습니다.")
        conn = psycopg2.connect(url)
        conn.autocommit = True
        _local.conn = conn
    return conn


def _build_tier_path(log: list, final_tier: int) -> str:
    """state['log']에서 에스컬레이션 경로를 추출한다."""
    path = [0]
    for line in log:
        if "Tier 1 에스컬레이션" in line and 1 not in path:
            path.append(1)
        if "Tier 2 에스컬레이션" in line and 2 not in path:
            path.append(2)
    return "→".join(str(t) for t in path)


def _build_row(
    state: GraphState,
    request_id: str,
    loop_number: int,
    is_final: bool,
    is_fallback: bool,
    execution_time_ms: Optional[int],
    final_answer: Optional[str],
    fk_grade: Optional[float] = None,
) -> dict:
    """공통 row dict 생성."""
    final_tier = state["search_tier"]
    ar = float(state.get("answer_relevance_score", 0.0) or 0.0)

    f_raw  = state.get("critic_score")
    cp_raw = state.get("context_precision_score")

    # Tier 1 평가: 논문 설계상 AR만 사용 → F, CP, q_total null.
    # - 중간행: graph.py _critic_node에서 tier==1 시 critic_score=None으로 전달 (f_raw is None)
    # - 최종행: is_final=True & final_tier==1 로 판별 (Tier 0→1 에스컬레이션 중간행 오적용 방지)
    if (is_final and final_tier == 1) or f_raw is None or cp_raw is None:
        f       = None
        cp      = None
        q_total = None
    else:
        f       = float(f_raw)
        cp      = float(cp_raw)
        q_total = round(0.4 * f + 0.4 * ar + 0.2 * cp, 6)

    hallu_flags = state.get("hallucination_flags") or []

    tier_path  = (state.get("tier_path") or "").strip()
    if not tier_path:
        tier_path = _build_tier_path(state.get("log", []), final_tier)
    is_escalated = (tier_path != "0")

    expected_tier_val = state.get("expected_tier", -1)
    expected_tier_db  = expected_tier_val if expected_tier_val >= 0 else None

    return {
        "request_id":             request_id,
        "loop_number":            loop_number,
        "is_final":               is_final,
        "ablation_condition":     state.get("ablation_condition") or None,
        "query_index":            state.get("query_index") or None,
        "disease":                state.get("disease") or None,
        "query_level_label":      state.get("query_level_label") or None,
        "user_level":             state.get("user_level") or "",
        "original_query":         state["question"],
        "optimized_query":        state["queries"][-1] if state.get("queries") else None,
        "expected_tier":          expected_tier_db,
        "final_tier":             final_tier,
        "tier_path":              tier_path,
        "is_escalated":           is_escalated,
        "is_fallback":            is_fallback,
        "self_correction_count":  int(state.get("self_correction_count", 0)),
        "ragas_f":                f,
        "ragas_ar":               ar,
        "ragas_cp":               cp,
        "q_total":                q_total,
        "hallucination_detected": len(hallu_flags) > 0,
        "hallucination_count":    len(hallu_flags),
        "retrieved_doc_count":    len(state.get("context") or []),
        "llm_model":              state.get("llm_provider") or "openai",
        "execution_time_ms":      execution_time_ms,
        "final_answer":           final_answer,
        "fk_grade":               fk_grade,
    }


def _execute_insert(row: dict) -> Optional[int]:
    """row dict를 INSERT하고 log_id를 반환한다."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(_INSERT_SQL, row)
            log_id: int = cur.fetchone()[0]
        def _fmt(v) -> str:
            return f"{v:.3f}" if v is not None else "null"
        logger.info(
            "[AuditLog] INSERT log_id=%d | rid=%s | loop=%d | is_final=%s | "
            "cond=%s | tier=%s | F=%s AR=%s CP=%s Q=%s",
            log_id, row["request_id"], row["loop_number"], row["is_final"],
            row["ablation_condition"], row["tier_path"],
            _fmt(row["ragas_f"]), _fmt(row["ragas_ar"]),
            _fmt(row["ragas_cp"]), _fmt(row["q_total"]),
        )
        return log_id
    except Exception as exc:
        logger.error("[AuditLog] INSERT 실패: %s", exc, exc_info=True)
        _local.conn = None
        return None


def save_loop_log(
    state: GraphState,
    request_id: str,
    loop_number: int,
) -> Optional[int]:
    """중간 루프 RAGAS 결과를 INSERT한다 (is_final=False).

    호출 시점: _critic_node에서 query_rewriter로 재시도/에스컬레이션할 때마다.
    final_answer와 execution_time_ms는 NULL로 저장된다.
    """
    user_level = state.get("user_level") or ""
    if user_level not in ("Professional", "Consumer"):
        logger.warning("[AuditLog] user_level=%r 유효하지 않음, loop_log 스킵.", user_level)
        return None

    row = _build_row(
        state=state,
        request_id=request_id,
        loop_number=loop_number,
        is_final=False,
        is_fallback=False,
        execution_time_ms=None,
        final_answer=None,
    )
    return _execute_insert(row)


def save_audit_log(
    state: GraphState,
    request_id: str,
    is_fallback: bool = False,
    execution_time_ms: Optional[int] = None,
    fk_grade: Optional[float] = None,
) -> Optional[int]:
    """최종 결과를 INSERT한다 (is_final=True).

    호출 시점: _output_node() 또는 _fallback_node() 완료 직후.
    fk_grade: graph.py에서 번역 전 영어 원문으로 계산해 전달한다.
    """
    user_level = state.get("user_level") or ""
    if user_level not in ("Professional", "Consumer"):
        logger.warning("[AuditLog] user_level=%r 유효하지 않음, audit_log 스킵.", user_level)
        return None

    if execution_time_ms is None:
        start = state.get("workflow_start_time")
        if start:
            execution_time_ms = int((time.time() - start) * 1000)

    loop_number = int(state.get("eval_count", 1))

    row = _build_row(
        state=state,
        request_id=request_id,
        loop_number=loop_number,
        is_final=True,
        is_fallback=is_fallback,
        execution_time_ms=execution_time_ms,
        final_answer=state.get("answer") or None,
        fk_grade=fk_grade,
    )
    return _execute_insert(row)
