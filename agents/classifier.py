from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.state import GraphState
from utils.json_parser import parse_llm_json, fallback_classifier_json
from core.llm_client import get_chat_llm, classifier_model


_CLASSIFIER_SYSTEM_PROMPT = """\
당신은 의료 정보 시스템의 사용자 수준 분류 전문가입니다.
사용자의 질문을 분석하여 질문자의 의료 지식 배경을 판단하세요.

[분류 기준]
- Professional(의료 전문가): 임상 용어·분자 기전 질의, 약물 기전·약동학, 진단 기준·감별 진단,
  처방 프로토콜, 검사 수치(HbA1c/eGFR/TSH 등) 해석, 병태생리, 치료 가이드라인 인용,
  영어 임상 용어 혼용, "기술하시오/설명하시오/비교하시오" 형 서술
- Consumer(일반인): 증상 설명, 복용 여부·부작용 문의, 생활 습관·식이 질문,
  쉬운 표현("~먹어도 되나요", "왜 그런가요", "어떻게 하나요", "알려주세요"),
  자가 진단·예방 정보 요청, 의학 용어 없이 일상 언어만 사용

[분류 예시]
Q: "인슐린 저항성(insulin resistance)의 분자적 기전(IRS-1 인산화, PI3K-Akt 경로 손상)과 췌장 베타세포의 점진적 기능 저하 사이의 병태생리학적 연관성을 설명하고, HbA1c 목표치 설정의 임상적 근거를 기술하시오."
→ {{"level":"Professional","reasoning":"분자 신호 경로와 HbA1c 임상 기준을 묻는 전문가 질의이다. 영어 임상 용어 혼용 및 '기술하시오' 형식이 특징적이다.","detected_intent":"기전_탐구","confidence":0.98}}

Q: "제2형 당뇨병은 왜 생기나요? 혈당 조절을 위해 식사할 때 특별히 주의해야 할 점은 무엇인가요?"
→ {{"level":"Consumer","reasoning":"원인과 식이 관리를 일상 언어로 묻는 일반인 질의이다. 의학 전문 용어 없이 쉬운 표현만 사용하고 있다.","detected_intent":"예방_정보","confidence":0.97}}

Q: "하시모토 갑상선염(Hashimoto's thyroiditis)에서 TPO 항체·갑상선글로불린 항체의 자가면역 기전과 레보티록신(levothyroxine) 용량 조절의 TSH 목표치 임상 기준을 기술하시오."
→ {{"level":"Professional","reasoning":"자가면역 기전과 검사 수치·처방 기준을 묻는 전문가 질의이다. 영어 임상 용어 혼용 및 '기술하시오' 형식이 나타난다.","detected_intent":"처방_결정","confidence":0.97}}

Q: "갑상선 약을 먹고 있는데 살이 잘 안 빠지고 추위를 많이 탑니다. 약을 계속 먹어도 되나요?"
→ {{"level":"Consumer","reasoning":"복용 중 증상과 복약 지속 여부를 일상 언어로 묻는 일반인 질의이다. 전문 용어 없이 개인 증상 경험을 서술하고 있다.","detected_intent":"복용법_확인","confidence":0.96}}

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
