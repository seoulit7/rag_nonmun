import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.state import GraphState
from core.llm_client import get_chat_llm, translate_model


_TRANSLATE_SYSTEM = (
    "You are a professional medical translator. "
    "The input may be in English, Chinese (Simplified or Traditional), "
    "Japanese, Korean, or a mix of these. "
    "Translate all substantive medical content into fluent Korean. "
    "Keep drug names, dosages, units, and lab names accurate; "
    "you may keep widely used international drug names in Latin script if standard. "
    "Do not refuse or summarize away content because the source is not English. "
    "Output only the Korean translation, no preamble or explanation."
)

_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _TRANSLATE_SYSTEM),
    ("human", "{text}"),
])


def _translate_to_korean(text: str) -> str:
    llm = get_chat_llm(model=translate_model(), temperature=0, max_tokens=4096)
    chain = _TRANSLATE_PROMPT | llm | StrOutputParser()
    translated = chain.invoke({"text": text}).strip()
    if not translated:
        raise RuntimeError("번역 결과가 비어 있습니다.")
    return translated


def _format_sources(sources: List[str]) -> str:
    """메타데이터 목록을 '파일명 p.N' 형태의 출처 문자열로 변환한다.
    파일이 여러 개면 개행으로 구분한다."""
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
    """한국어 번역 및 출처·면책 조항 추가 에이전트."""
    korean_answer = _translate_to_korean(state["answer"])
    state["log"].append("[Output] 다국어 원문 -> 한국어 번역 완료.")

    sources = state.get("context_sources", [])
    tier = int(state.get("search_tier", 0))

    if tier == 1:
        is_gemini = state.get("llm_provider") == "gemini"
        backend = "Gemini" if is_gemini else "GPT"
        source_line = f"출처: LLM 학습데이터 ({backend})"
        disclaimer = (
            f"이 정보는 {backend}가 학습한 일반 지식에 근거한 설명입니다. "
            "진료에 대한 최종 판단은 전문 의료진과 상의하십시오."
        )
        log_src = source_line
    elif tier == 2:
        source_detail = _format_sources(sources) if sources else "웹 검색"
        multiline = "\n" in source_detail
        source_line = (
            f"출처: 웹 검색\n{source_detail}" if multiline
            else f"출처: 웹 검색 - {source_detail}"
        )
        disclaimer = (
            "이 정보는 공개 웹 검색 결과를 바탕으로 생성되었습니다. "
            "출처의 신뢰도를 확인하시고, 진료에 대한 최종 판단은 전문 의료진과 상의하십시오."
        )
        log_src = source_detail
    else:
        source_detail = _format_sources(sources) if sources else "MSD 매뉴얼"
        multiline = "\n" in source_detail
        source_line = (
            f"출처: MSD 매뉴얼\n{source_detail}" if multiline
            else f"출처: MSD 매뉴얼 - {source_detail}"
        )
        disclaimer = (
            "이 정보는 MSD 매뉴얼을 기반으로 생성된 답변입니다. "
            "진료에 대한 최종 판단은 전문 의료진과 상의하십시오."
        )
        log_src = source_detail

    state["answer"] = f"{korean_answer}\n\n{source_line}\n\n{disclaimer}"
    state["log"].append(f"[Output] 출처: {log_src}")
    state["log"].append("[Output] 최종 답변에 출처 및 면책 조항 추가 완료.")

    return state
