import os
from typing import List

from models.state import GraphState


def _format_sources(sources: List[str]) -> str:
    seen = []
    for src in sources:
        if "#p" in src:
            path_part, page_part = src.rsplit("#p", 1)
            entry = f"{os.path.basename(path_part)} p.{int(page_part) + 1}"
        else:
            entry = os.path.basename(src)
        if entry not in seen:
            seen.append(entry)
    if len(seen) <= 1:
        return seen[0] if seen else ""
    return "\n".join(f"  • {e}" for e in seen)


def output_agent(state: GraphState) -> GraphState:
    """Append source citation and disclaimer to the English answer."""
    answer = state["answer"]
    sources = state.get("context_sources", [])
    tier = int(state.get("search_tier", 0))

    if tier == 1:
        is_gemini = state.get("llm_provider") == "gemini"
        backend = "Gemini" if is_gemini else "GPT"
        source_line = f"Source: LLM training data ({backend})"
        disclaimer = (
            f"This information is based on general medical knowledge from {backend}. "
            "For clinical decisions, please consult a qualified healthcare professional."
        )
        log_src = source_line
    elif tier == 2:
        source_detail = _format_sources(sources) if sources else "web search"
        multiline = "\n" in source_detail
        source_line = (
            f"Source: Web Search\n{source_detail}" if multiline
            else f"Source: Web Search - {source_detail}"
        )
        disclaimer = (
            "This information is based on publicly available web search results. "
            "Please verify the source reliability and consult a qualified healthcare professional for clinical decisions."
        )
        log_src = source_detail
    else:
        source_detail = _format_sources(sources) if sources else "MSD Manual"
        multiline = "\n" in source_detail
        source_line = (
            f"Source: MSD Manual\n{source_detail}" if multiline
            else f"Source: MSD Manual - {source_detail}"
        )
        disclaimer = (
            "This answer is generated based on the MSD Manual. "
            "For clinical decisions, please consult a qualified healthcare professional."
        )
        log_src = source_detail

    user_level = state.get("user_level", "")
    if user_level in ("Professional", "Consumer"):
        answer = f"[{user_level} Summary] {answer}"
    state["answer"] = f"{answer}\n\n{source_line}\n\n{disclaimer}"
    state["log"].append(f"[Output] Source: {log_src}")
    state["log"].append("[Output] Final answer with source and disclaimer appended.")

    return state
