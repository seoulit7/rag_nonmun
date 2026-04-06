from typing import Dict

import streamlit as st

from ui.utils import score_label


def render_score_card(scores: Dict[str, float]) -> None:
    """제출 완료 후 RAGAS 최종 품질 지표 3-컬럼 카드를 렌더링한다."""
    st.markdown("#### RAGAS 최종 품질 지표")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Faithfulness",
        f"{scores['faithfulness']:.2f}",
        score_label(scores["faithfulness"]),
        delta_color="normal" if scores["faithfulness"] >= 0.8 else "inverse",
    )
    c2.metric(
        "Answer Relevance",
        f"{scores['answer_relevance']:.2f}",
        score_label(scores["answer_relevance"]),
        delta_color="normal" if scores["answer_relevance"] >= 0.8 else "inverse",
    )
    c3.metric(
        "Context Precision",
        f"{scores['context_precision']:.2f}",
        score_label(scores["context_precision"]),
        delta_color="normal" if scores["context_precision"] >= 0.8 else "inverse",
    )
    st.markdown("---")
