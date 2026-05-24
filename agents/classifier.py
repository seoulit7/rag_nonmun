from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.state import GraphState
from utils.json_parser import parse_llm_json, fallback_classifier_json
from core.llm_client import get_chat_llm, classifier_model


_CLASSIFIER_SYSTEM_PROMPT = """\
당신은 의료 정보 시스템의 사용자 수준 분류 전문가입니다.
사용자의 질문을 분석하여 질문자의 의료 지식 배경을 판단하세요.

[분류 기준]
- Professional(의료 전문가): 임상 용어 사용, 약물 기전·약동학 질문, 진단 기준·감별 진단,
  처방 프로토콜, 검사 수치(HbA1c/eGFR 등) 해석, 병리 기전, 치료 가이드라인 참조
- Consumer(일반인): 증상 설명, 복용 여부 문의, 부작용 경험 공유,
  생활 속 건강 질문, 쉬운 표현("~먹어도 되나요", "이게 뭔가요"), 자가 진단 시도

[detected_intent 후보]
부작용_문의 / 복용법_확인 / 진단_기준 / 처방_결정 / 증상_설명 /
기전_탐구 / 예방_정보 / 검사_해석 / 약물_상호작용 / 기타

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요:
{{
  "level": "Professional 또는 Consumer",
  "confidence": 0.0~1.0,
  "reasoning": "분류 근거 (한국어, 2문장 이내)",
  "detected_intent": "위 후보 중 하나"
}}"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CLASSIFIER_SYSTEM_PROMPT),
    ("human", "분류할 질문: {question}"),
])


def _classify_with_llm(question: str) -> dict:
    llm = get_chat_llm(model=classifier_model(), temperature=0.1, max_tokens=1024)
    chain = _PROMPT | llm.bind(response_format={"type": "json_object"}) | StrOutputParser()
    raw = chain.invoke({"question": question})
    data = parse_llm_json(raw)
    if not data.get("level"):
        data.update(fallback_classifier_json(raw))
    return data


def level_classifier(state: GraphState) -> GraphState:
    """LLM 기반 사용자 수준 분류 에이전트. 이미 user_level이 설정된 경우 스킵."""
    if state.get("user_level"):
        return state

    result = _classify_with_llm(state["question"])

    level = result.get("level", "Consumer")
    if level not in ("Professional", "Consumer"):
        level = "Consumer"

    confidence: float = float(result.get("confidence", 0.0))
    reasoning: str = result.get("reasoning", "")
    intent: str = result.get("detected_intent", "기타")

    state["user_level"] = level
    state["log"].append(
        f"[Level] LLM 분류: {level} "
        f"(신뢰도={confidence:.2f}, 의도={intent})"
    )
    if reasoning:
        state["log"].append(f"[Level] 근거: {reasoning}")

    return state
