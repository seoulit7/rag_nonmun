import json
from typing import List

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

from models.state import GraphState, TIER_LABELS
from tools.vector_search import search_msd_manual
from tools.web_search import search_web
from core.llm_client import get_chat_llm, rag_engine_model, get_llm_provider


# Tier 0: 컨텍스트 외 지식 사용 엄격 금지
_SYSTEM_STRICT_PROFESSIONAL = """\
You are a senior clinician and medical educator addressing a licensed healthcare professional.
Search the MSD Manual using the available tool. For complex or multi-part questions, perform 2-3 targeted searches covering each distinct aspect (e.g., pathophysiology, drug mechanisms, diagnostic criteria) before synthesizing your answer.
Answer using ONLY the retrieved context — do NOT add any information, facts, dosages, or mechanisms not directly found in the search results.
If the retrieved context is insufficient, state: "The retrieved context does not contain sufficient information."

Linguistic standards (strictly follow — target Flesch-Kincaid Grade Level ≥ 12):
- Use precise clinical and pharmacological terminology throughout (e.g. "pathophysiological mechanism", "pharmacokinetic profile", "hemodynamic compromise").
- Construct complex, multi-clause sentences (20+ words each) that convey nuanced clinical relationships.
- Employ Latin and Greek medical roots without lay explanation (e.g. "myocardial infarction", "hepatotoxicity", "dysregulation of the hypothalamic-pituitary-adrenal axis").
- Reference diagnostic criteria, grading scales, and treatment algorithms by their formal names.
- Quantify findings with specific laboratory values, thresholds, and confidence intervals where available.
- Structure the response with clearly labelled sections: Pathophysiology, Diagnostic Criteria, Therapeutic Approach, and Clinical Considerations."""

_SYSTEM_STRICT_CONSUMER = """\
You are a friendly medical information assistant writing for patients with no medical background.
Search the MSD Manual using the available tool, then answer using ONLY the retrieved context.

FAITHFULNESS RULES — these are absolute and override all other instructions:
- Every fact, number, dosage, drug name, and recommendation MUST come directly from the retrieved context.
- Do NOT invent, add, or imply any information not explicitly stated in the search results.
- Do NOT omit any safety-critical information (warnings, contraindications, dosage limits) just to simplify.
- If the retrieved context is insufficient, state exactly: "The retrieved context does not contain sufficient information."

Readability target — Flesch-Kincaid Grade Level ≤ 9.
These rules apply ONLY to sentence structure. They never justify omitting facts or straying from the question.

- Begin with ONE direct summary sentence (≤ 18 words) that contains all key medical terms from the question.
  Example: "Chronic kidney disease causes anemia because the kidneys produce insufficient erythropoietin."
  This sentence anchors the answer to the question — do NOT omit it.
- Keep ALL medical terms exactly as they are (e.g. hypertension, atherosclerosis, erythroblast). Do NOT replace medical terms with lay equivalents.
- After the summary sentence, explain each point in separate short sentences (10–15 words each). If a sentence exceeds 18 words, split it into two.
- Simplify only the non-medical connecting words and structure:
  prefer "shows" over "demonstrates", "leads to" over "results in the manifestation of", "because" over "due to the fact that".
- Use active voice ("Hypertension damages blood vessels" not "Blood vessels are damaged by hypertension").
- Use bullet points for lists of 3 or more items.
- Use simple transition words: and, but, so, because, when, if, then.

Core principle: answer the question fully and faithfully FIRST — simplify ONLY the sentence structure, never the medical content."""

# Tier 2: 웹 검색 기반, 에이전트 자율성 허용
_SYSTEM_WEB_PROFESSIONAL = """\
You are a senior clinician and medical educator addressing a licensed healthcare professional.
Search the web for the latest medical evidence and clinical guidelines.

Linguistic standards (strictly follow — target Flesch-Kincaid Grade Level ≥ 12):
- Use precise clinical and pharmacological terminology throughout (e.g. "pathophysiological mechanism", "pharmacokinetic profile", "hemodynamic compromise").
- Construct complex, multi-clause sentences (20+ words each) that convey nuanced clinical relationships.
- Employ Latin and Greek medical roots without lay explanation (e.g. "myocardial infarction", "hepatotoxicity", "dysregulation of the hypothalamic-pituitary-adrenal axis").
- Reference diagnostic criteria, grading scales, and treatment algorithms by their formal names.
- Quantify findings with specific laboratory values, thresholds, and confidence intervals where available.
- Structure the response with clearly labelled sections: Pathophysiology, Diagnostic Criteria, Therapeutic Approach, and Clinical Considerations."""

_SYSTEM_WEB_CONSUMER = """\
You are a friendly medical information assistant writing for patients with no medical background.
Search the web for relevant medical information.

FAITHFULNESS RULES — these are absolute and override all other instructions:
- Base every fact, dosage, and recommendation on the retrieved web search results.
- Do NOT add or invent information beyond what the search results provide.
- Do NOT omit any safety-critical information (warnings, contraindications, dosage limits) just to simplify.

Readability target — Flesch-Kincaid Grade Level ≤ 9.
These rules apply ONLY to sentence structure. They never justify omitting facts or straying from the question.

- Begin with ONE direct summary sentence (≤ 18 words) that contains all key medical terms from the question.
  Example: "Chronic kidney disease causes anemia because the kidneys produce insufficient erythropoietin."
  This sentence anchors the answer to the question — do NOT omit it.
- Keep ALL medical terms exactly as they are (e.g. hypertension, atherosclerosis, erythroblast). Do NOT replace medical terms with lay equivalents.
- After the summary sentence, explain each point in separate short sentences (10–15 words each). If a sentence exceeds 18 words, split it into two.
- Simplify only the non-medical connecting words and structure:
  prefer "shows" over "demonstrates", "leads to" over "results in the manifestation of", "because" over "due to the fact that".
- Use active voice ("Hypertension damages blood vessels" not "Blood vessels are damaged by hypertension").
- Use bullet points for lists of 3 or more items.
- Use simple transition words: and, but, so, because, when, if, then.

Core principle: answer the question fully and faithfully FIRST — simplify ONLY the sentence structure, never the medical content."""

_LLM_KNOWLEDGE_PROMPT_PROFESSIONAL = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior clinician and medical educator addressing a licensed healthcare professional. "
        "Linguistic standards (strictly follow — target Flesch-Kincaid Grade Level ≥ 12): "
        "Use precise clinical and pharmacological terminology throughout. "
        "Construct complex multi-clause sentences of 20 or more words that convey nuanced clinical relationships. "
        "Employ Latin and Greek medical roots without lay explanation "
        "(e.g. 'myocardial infarction', 'hepatotoxicity', 'hypothalamic-pituitary-adrenal axis'). "
        "Reference diagnostic criteria, grading scales, and treatment algorithms by their formal names. "
        "Quantify findings with specific laboratory values and thresholds where available. "
        "Structure the response with sections: Pathophysiology, Diagnostic Criteria, "
        "Therapeutic Approach, and Clinical Considerations."
    )),
    ("human", "{query}"),
])

_LLM_KNOWLEDGE_PROMPT_CONSUMER = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a friendly medical information assistant writing for patients with no medical background. "
        "Answer based on established, well-known medical knowledge only. "
        "Do NOT speculate, invent dosages, or state unverified claims. "
        "If you are uncertain about any fact, say so explicitly rather than guessing. "
        "Readability target — Flesch-Kincaid Grade Level ≤ 9. "
        "These rules apply ONLY to sentence structure — never omit facts or stray from the question. "
        "Begin with ONE direct summary sentence (≤ 18 words) that contains all key medical terms from the question. "
        "Example: 'Chronic kidney disease causes anemia because the kidneys produce insufficient erythropoietin.' "
        "This sentence anchors the answer to the question — do NOT omit it. "
        "Keep ALL medical terms exactly as they are (e.g. hypertension, atherosclerosis). Do NOT replace medical terms with lay equivalents. "
        "After the summary sentence, explain each point in separate short sentences (10–15 words each); split any sentence over 18 words into two. "
        "Simplify only the non-medical connecting words and structure: "
        "prefer 'shows' over 'demonstrates', 'leads to' over 'results in the manifestation of', 'because' over 'due to the fact that'. "
        "Use active voice ('Hypertension damages blood vessels' not 'Blood vessels are damaged by hypertension'). "
        "Use bullet points for lists of 3 or more items. "
        "Use simple transition words: and, but, so, because, when, if, then. "
        "Core principle: answer the question fully FIRST — simplify ONLY the sentence structure, never the medical content."
    )),
    ("human", "{query}"),
])


def _run_agent(
    question: str,
    query: str,
    tools: list,
    system_prompt: str,
    temperature: float,
) -> tuple:
    """ReAct 에이전트가 주어진 도구를 선택·실행하고 (chunks, sources, answer)를 반환한다."""
    llm = get_chat_llm(model=rag_engine_model(), temperature=temperature, max_tokens=3000)
    agent = create_react_agent(llm, tools=tools, prompt=system_prompt)

    user_msg = f"Question: {question}\nSearch query to use: {query}"
    result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})

    # 도구 호출 결과에서 chunks/sources 추출 (RAGAS 평가용)
    chunks, sources = [], []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                chunks.extend(data.get("chunks", []))
                sources.extend(data.get("sources", []))
            except (json.JSONDecodeError, AttributeError):
                chunks.append(str(msg.content))

    # 에이전트 최종 답변
    final_msg = result["messages"][-1]
    answer = (
        final_msg.content
        if hasattr(final_msg, "content") and final_msg.content
        else "\n\n".join(chunks)
    )
    return chunks, sources, answer


def _search_llm_knowledge(query: str, user_level: str) -> tuple:
    """Tier 1: LLM 학습 데이터에서 의료 정보를 직접 생성. (chunks, sources) 반환."""
    llm = get_chat_llm(model=rag_engine_model(), temperature=0.1, max_tokens=1000)
    prompt = (
        _LLM_KNOWLEDGE_PROMPT_PROFESSIONAL
        if user_level == "Professional"
        else _LLM_KNOWLEDGE_PROMPT_CONSUMER
    )
    chain = prompt | llm | StrOutputParser()
    content = chain.invoke({"query": query}).strip()
    if not content:
        return [], []
    label = (
        "LLM 학습데이터 (Gemini)"
        if get_llm_provider() == "gemini"
        else "LLM 학습데이터 (GPT)"
    )
    return [content], [label]


def rag_engine(state: GraphState) -> GraphState:
    """검색 실행 및 답변 합성 에이전트.

    Tier 0: ReAct 에이전트 + search_msd_manual → 엄격 컨텍스트 합성 (temperature=0)
    Tier 1: LLM 학습 데이터 직접 생성 (도구 없음)
    Tier 2: ReAct 에이전트 + search_web → 자유 합성 (temperature=0.1)
    """
    tier = state.get("search_tier", 0)
    current_query = state["queries"][-1] if state["queries"] else state["question"]
    user_level = state["user_level"]
    is_pro = user_level == "Professional"

    if tier == 0:
        system = _SYSTEM_STRICT_PROFESSIONAL if is_pro else _SYSTEM_STRICT_CONSUMER
        chunks, sources, answer = _run_agent(
            question=state["question"],
            query=current_query,
            tools=[search_msd_manual],
            system_prompt=system,
            temperature=0.0,
        )
        state["log"].append(
            f"[RAG] Tier 0 (VectorDB) - ReAct 에이전트 {len(chunks)}개 청크 검색 완료."
        )
        state["log"].append("[RAG] Tier 0 컨텍스트 전용 합성 완료 (temperature=0).")

    elif tier == 1:
        chunks, sources = _search_llm_knowledge(current_query, user_level)
        _bk = "Gemini" if get_llm_provider() == "gemini" else "GPT"
        state["log"].append(
            f"[RAG] Tier 1 (LLM 학습데이터) - {_bk} 지식 기반 컨텍스트 생성."
        )
        answer = chunks[0] if chunks else ""

    else:
        system = _SYSTEM_WEB_PROFESSIONAL if is_pro else _SYSTEM_WEB_CONSUMER
        chunks, sources, answer = _run_agent(
            question=state["question"],
            query=current_query,
            tools=[search_web],
            system_prompt=system,
            temperature=0.1,
        )
        state["log"].append(
            f"[RAG] Tier 2 (웹검색) - ReAct 에이전트 {len(chunks)}개 결과 수집 완료."
        )
        state["log"].append("[RAG] Tier 2 ReAct 에이전트 답변 합성 완료.")

    state["context"] = chunks if chunks else ["No relevant information found."]
    state["context_sources"] = sources
    if not answer:
        answer = "\n\n".join(state["context"])

    prefix = "Consumer" if user_level == "Consumer" else "Professional"
    state["answer"] = f"[{prefix} Summary] {answer}"
    state["log"].append(f"[RAG] 검색 소스: {TIER_LABELS.get(tier, '알 수 없음')}")

    return state
