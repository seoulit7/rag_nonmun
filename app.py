import logging

import streamlit as st

import config.settings as settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
settings.setup_logging()
from medical_rag_graph import run_medical_self_corrective_rag
from ui import (
    SESSION_DEFAULTS,
    render_sidebar,
    render_header,
    render_pdf_uploader,
    on_step,
    render_score_card,
    render_result,
    render_log,
)

st.set_page_config(page_title="Medical Self-Corrective RAG", layout="wide")

for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── 앱 시작 시 FAISS 인덱스 자동 초기화 ──────────────────────────────────────
# Streamlit Cloud는 재시작마다 파일시스템이 초기화되므로 매번 빌드 필요.
# session_state로 중복 실행 방지 (같은 세션에서는 한 번만 실행).
if not st.session_state.get("_vector_db_initialized"):
    from pathlib import Path
    from tools.vector_search import initialize_vector_db
    index_exists = Path(settings.INDEX_PATH).exists()
    if not index_exists:
        with st.spinner("FAISS 인덱스 초기화 중... (최초 실행 시 수 분 소요)"):
            try:
                initialize_vector_db()
                st.toast("인덱스 초기화 완료!", icon="✅")
            except Exception as e:
                st.warning(f"인덱스 초기화 실패: {e}")
    st.session_state["_vector_db_initialized"] = True

user_persona, llm_backend, dashboard_menu = render_sidebar()

# ── 대시보드 메뉴 선택 시 해당 화면만 표시 ────────────────────────────────────
if dashboard_menu == "로그 조회":
    from ui.dashboard import render_log_viewer
    render_log_viewer()
    st.stop()

elif dashboard_menu == "성능 시각화":
    from ui.dashboard import render_performance_viz
    render_performance_viz()
    st.stop()

# ── 기본 RAG 화면 ─────────────────────────────────────────────────────────────
render_pdf_uploader()
render_header()

question = st.text_area("문의할 내용을 입력하세요:", height=120)

if st.button("질문 제출", type="primary"):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        forced = (
            None
            if user_persona == "자동 분류"
            else ("Professional" if user_persona == "의료 전문가" else "Consumer")
        )
        prov = "gemini" if llm_backend == "Gemini" else "openai"

        for key, default in SESSION_DEFAULTS.items():
            st.session_state[key] = default

        if prov == "gemini" and not settings.get_gemini_api_key().strip():
            st.warning("Gemini를 사용하려면 .env에 GEMINI_API_KEY를 설정하세요.")
        else:
            try:
                with st.status("⚙️ Self-Corrective RAG 실행 중...", expanded=True) as status:
                    final_state = run_medical_self_corrective_rag(
                        question,
                        forced_user_level=forced,
                        step_callback=on_step,
                        llm_provider=prov,
                    )
                    had_fallback = not (
                        final_state.get("critic_score", 0.0) >= settings.FAITHFULNESS_THRESHOLD
                    ) or any("최대 재시도" in l for l in final_state.get("log", []))
                    status.update(
                        label="⚠️ 분석 완료 (신뢰도 부족)" if had_fallback else "✅ 분석 완료!",
                        state="error" if had_fallback else "complete",
                        expanded=False,
                    )

                st.session_state.logs = final_state["log"]
                st.session_state.result = final_state["answer"]
                st.session_state.detected_level = final_state["user_level"]
                st.session_state.search_tier = final_state.get("search_tier", 0)
                st.session_state.llm_provider = final_state.get("llm_provider", prov)
                st.session_state.scores = {
                    "faithfulness": final_state.get("critic_score", 0.0),
                    "answer_relevance": final_state.get("answer_relevance_score", 0.0),
                    "context_precision": final_state.get("context_precision_score", 0.0),
                }
                st.rerun()

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

if st.session_state.scores:
    render_score_card(st.session_state.scores)

if st.session_state.result:
    render_result(
        st.session_state.result,
        st.session_state.get("search_tier", 0),
        st.session_state.get("llm_provider") or "openai",
    )

if st.session_state.logs:
    render_log(st.session_state.logs)
