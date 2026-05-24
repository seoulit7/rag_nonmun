from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.state import GraphState
from utils.json_parser import parse_llm_json, fallback_classifier_json
from core.llm_client import get_chat_llm, classifier_model


_CLASSIFIER_SYSTEM_PROMPT = """\
당신은 의료 정보 시스템의 사용자 수준 분류 전문가입니다.
사용자의 질문을 분석하여 질문자의 의료 지식 배경을 판단하세요.

[분류 기준]
- Professional(의료 전문가): 약물 기전·약동학 질문, 진단 기준·감별 진단, 처방 프로토콜,
  검사 수치(HbA1c/eGFR 등) 해석, 분자·세포 수준 병태생리, 치료 가이드라인 참조,
  "기술하시오/설명하시오/비교하시오" 형 서술형 지문
- Consumer(일반인): 증상 설명, 복용 여부·부작용 문의, 생활 습관·식이 질문,
  "~인가요?/~나요?/~알려주세요/~궁금합니다" 형 일상 문체,
  질병 원인·예방 등 일반 건강 궁금증 (의학 용어를 병기해도 무관)

[핵심 판별 원칙]
- "왜 ~인가요?", "어떤 증상이 나타나나요?", "어떻게 대처해야 하나요?" 형식은
  의학 용어나 영어 병기가 있더라도 Consumer로 분류한다.
- Professional은 임상 전문가만 관심을 갖는 내용(약동학, 분자 기전, 검사 수치 해석,
  처방 프로토콜)을 직접 묻는 경우에만 해당한다.

[분류 예시]
Q: "고혈압이 있으면 왜 뇌졸중 위험이 높아지나요?"
→ {{"level":"Consumer","reasoning":"일상 언어로 고혈압과 뇌졸중의 관계를 묻는 일반인 질의이다. '왜~나요?' 형식은 예방 정보를 구하는 일반인의 전형적 표현이다.","detected_intent":"예방_정보","confidence":0.95}}

Q: "본태성 고혈압에서 RAAS의 활성화 기전을 설명하고 ACEI와 ARB의 약리학적 작용 차이를 비교하시오."
→ {{"level":"Professional","reasoning":"RAAS 기전과 약리학적 비교를 요구하는 전문가 지문이다. '비교하시오' 서술형 형식과 약동학 내용이 특징적이다.","detected_intent":"기전_탐구","confidence":0.97}}

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
