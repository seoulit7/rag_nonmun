import logging

import streamlit as st

import config.settings as settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
settings.setup_logging()
from graph import run_medical_self_corrective_rag
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


user_persona, llm_backend, dashboard_menu = render_sidebar()

if dashboard_menu == "Log Query":
    from ui.dashboard import render_log_viewer
    render_log_viewer()
    st.stop()

elif dashboard_menu == "Performance Visualization":
    from ui.dashboard import render_performance_viz
    render_performance_viz()
    st.stop()

render_pdf_uploader()
render_header()

question = st.text_area("Enter your medical question:", height=120)

if st.button("Submit Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        forced = (
            None
            if user_persona == "Auto Detect"
            else ("Professional" if user_persona == "Medical Professional" else "Consumer")
        )
        prov = "openai"

        for key, default in SESSION_DEFAULTS.items():
            st.session_state[key] = default

        try:
            with st.status("⚙️ Running Self-Corrective RAG...", expanded=True) as status:
                    final_state = run_medical_self_corrective_rag(
                        question,
                        forced_user_level=forced,
                        step_callback=on_step,
                        llm_provider=prov,
                        ablation_condition="A",
                    )
                    had_fallback = not (
                        final_state.get("critic_score", 0.0) >= settings.FAITHFULNESS_THRESHOLD
                    ) or any(
                        "max retry" in l.lower() or "최대 재시도" in l
                        for l in final_state.get("log", [])
                    )
                    status.update(
                        label="⚠️ Analysis complete (low confidence)" if had_fallback else "✅ Analysis complete!",
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
            st.error(f"An error occurred: {e}")

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
