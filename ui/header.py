import streamlit as st

from ui.constants import PNG_PATH, SVG_PATH

_PIPELINE_DESCRIPTION = """
---
#### Pipeline Step-by-Step Description

**1. level_classifier `Agent` — User Level Classification**
The LLM analyzes the question's writing style and terminology to classify the user as either
`Professional` (medical expert) or `Consumer` (general public).
All subsequent query optimization and answer generation styles follow this classification.

---

**2. adaptive_query_rewriter `Agent` — Query Optimization / Rewriting**
- **First run**: Converts the user question into an optimized English query for MSD Manual search.
- **On retry/escalation**: Provides previous evaluation results (F·AR·CP scores) to the LLM to generate a more accurate query.

---

**3. rag_engine `Agent` — Search Execution and Answer Synthesis**
This agent directly calls the appropriate search function by Tier and synthesizes results using an LLM.
The Tier determines which tool is used; the LLM generates the answer from the collected context.

```
rag_engine  ← LangGraph node
  ├─ Tier 0 → search_vector_db(query)  direct call (FAISS)        ← Tool
  ├─ Tier 1 → _search_llm_knowledge()  direct call (no external tool)
  └─ Tier 2 → search_web(query)        direct call (DuckDuckGo)   ← Tool
```

Each tool call result (context) is combined with the user question and synthesized into the final answer.
(Tier 1 skips synthesis as `_search_llm_knowledge` already returns LLM-generated text.)

---

**4. Tier 0 — VectorDB (FAISS)** · `search_vector_db` `Tool`
Performs semantic search on the MSD Manual FAISS index to retrieve relevant document chunks.

| Post-evaluation branch | Condition | Action |
|---|---|---|
| Pass | F ≥ 0.8 | Move to output_agent |
| Retry | F < 0.8, retries < 3, not critical | Refine query and re-search |
| Immediate escalation | **AR < 0.3** (no relevant content in VectorDB) | Move immediately to Tier 1 |
| Immediate escalation | **F < 0.3 AND CP < 0.2** (search completely off-target) | Move immediately to Tier 1 |
| Exhausted escalation | After 3 retries, still below threshold | Move to Tier 1 |

> **Immediate escalation**: If AR < 0.3, the VectorDB is deemed to lack the relevant content.
> No amount of query refinement can find content that doesn't exist — escalates directly to Tier 1 without retrying.

---

**5. Tier 1 — LLM Knowledge**
Directly utilizes the pre-trained knowledge of the LLM (GPT / Gemini) without external search.
One attempt: outputs if F ≥ 0.8, otherwise escalates to Tier 2.

---

**6. Tier 2 — Web Search (DuckDuckGo)** · `search_web` `Tool`
Retrieves up-to-date medical information from the internet.
The retrieved context and user question are combined as a prompt and synthesized via LLM.
One attempt: outputs if F ≥ 0.8, otherwise moves to fallback.

---

**7. critic_agent `Agent` — RAGAS Quality Evaluation**
An LLM-based RAGAS evaluator computing three metrics.

| Metric | Meaning | Threshold |
|---|---|---|
| **Faithfulness (F)** | Is the answer grounded in the retrieved context? | ≥ 0.8 to pass |
| **Answer Relevance (AR)** | How relevant is the answer to the question? | < 0.3 triggers immediate escalation |
| **Context Precision (CP)** | How precisely relevant are the retrieved chunks? | F<0.3 & CP<0.2 triggers immediate escalation |

---

**8. output_agent `Agent` — Final Output**
Appends source citations (MSD Manual file·page / LLM / web URL) and a medical disclaimer to the English answer.

---

**9. fallback — Raw Context Presentation**
If F < 0.8 after exhausting all Tiers, a reliable answer could not be generated.
The raw retrieved source material is presented as-is.
"""


def render_header() -> None:
    """Render the main title, pipeline expander, and download buttons."""
    st.title("Medical Self-Corrective RAG")
    st.markdown(
        "Question → Level Classification → Query Optimization → Search (VectorDB→LLM→Web) → RAGAS Evaluation → Self-Correction → English Answer"
    )

    with st.expander("LangGraph Pipeline Visualization & Description"):
        if PNG_PATH:
            st.image(str(PNG_PATH), width="stretch")
        else:
            st.warning("No PNG file found in the image/ folder.")
        st.markdown(_PIPELINE_DESCRIPTION)

    col_png, col_svg, _ = st.columns([1, 1, 4])
    if PNG_PATH:
        col_png.download_button(
            label="⬇ Download PNG",
            data=PNG_PATH.read_bytes(),
            file_name=PNG_PATH.name,
            mime="image/png",
        )
    if SVG_PATH:
        col_svg.download_button(
            label="⬇ Download SVG",
            data=SVG_PATH.read_bytes(),
            file_name=SVG_PATH.name,
            mime="image/svg+xml",
        )
