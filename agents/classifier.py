import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.state import GraphState
from utils.json_parser import parse_llm_json, fallback_classifier_json
from core.llm_client import get_chat_llm, classifier_model


# ── 규칙 기반 전문가 지문 (LLM 오분류 방지) ────────────────────────────────────
_PROFESSIONAL_RULE = re.compile(
    r"(?:기술하시오|설명하시오|요약하시오|비교하여\s*(?:설명하시오|기술하시오)"
    r"|분류하여\s*기술(?:하시오)?)",
)

# ── 규칙 기반 Consumer 사전 분류 (LLM 오분류 방지) ──────────────────────────────
_CONSUMER_RULE = re.compile(
    r'(?:'
    # 한국어 일상 문체 패턴
    r'이유[는가이]\s*(?:무엇|뭔)'      # "이유는/가/이 무엇인가요?"
    r'|왜\s+.{1,30}?(?:나요|인가요|됩니까|합니까)\??'  # "왜 ~나요/인가요?"
    r'|오래\s+방치'                     # "오래 방치하면"
    r'|어떻게\s+(?:다른|구별|대처|알)'
    r'|한\s*가지(?:만)?'
    r'|알려\s*달라'
    r'|피해야\s+하는'                   # "피해야 하는 이유"
    r'|알려주세요'
    r'|설명해주세요'
    r'|궁금합니다'
    r'|무엇이\s+있나요'
    r'|어떤\s+증상'
    r'|가야\s+하나요'
    # 영어 환자 관점 문체 패턴 (P 질문은 Patients/According to/In [clinical context]로 시작)
    r'|^People\s+(?:with|who|in|whose)\b'
    r'|^If\s+(?:you|someone|a\s+(?:man|woman|person))\s+(?:have|has|had|get|gets|want)\b'
    r'|^For\s+(?:someone|people)\s+with\b'
    r'|^Everyone\s+feels?\b'
    r'|^Women\s+with\b'
    r'|^A\s+(?:person|man|woman)\s+with\b'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


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
- "~이유는 무엇인가요?", "오래 방치하면 왜 ~이 생기나요?", "~을 피해야 하는 이유"
  형식도 일반인의 건강 정보 요구이므로 Consumer로 분류한다.
- 질환명(COPD, 위식도역류, 갑상선기능저하증, 만성신장질환 등)이 포함되어 있어도
  일상 언어로 원인·예방·합병증을 묻는다면 반드시 Consumer로 분류한다.
- Professional은 임상 전문가만 관심을 갖는 내용(약동학, 분자 기전, 검사 수치 해석,
  처방 프로토콜, "~설명하시오/기술하시오/비교하시오" 서술형 지문)에만 해당한다.
- [영어 질문 추가 원칙] "According to the MSD Manual"을 인용하는 영어 질문도
  환자 교육·질환 이해·예방 정보 목적이면 Consumer이다. MSD Manual 인용 자체는
  Professional 분류의 근거가 아니다.
- [영어 질문 추가 원칙] 영어 질문에서 질병명·증상명으로 시작하더라도 증상·원인·
  합병증·치료 결과를 일반인 관점에서 묻는다면 Consumer이다. Professional은
  처방 결정·약동학·감별 진단 프로토콜·약물 기전 등 임상 실무에만 해당한다.

[분류 예시]
Q: "고혈압이 있으면 왜 뇌졸중 위험이 높아지나요?"
→ {{"level":"Consumer","reasoning":"일상 언어로 고혈압과 뇌졸중의 관계를 묻는 일반인 질의이다. '왜~나요?' 형식은 예방 정보를 구하는 일반인의 전형적 표현이다.","detected_intent":"예방_정보","confidence":0.95}}

Q: "흡연이 만성 폐쇄성 폐질환(COPD)을 유발하는 이유는 무엇인가요?"
→ {{"level":"Consumer","reasoning":"COPD라는 의학 용어가 있지만 '이유는 무엇인가요?' 형식은 일반인이 질환 원인을 궁금해하는 전형적 패턴이다. 흡연과 질환의 관계를 일상 언어로 묻고 있어 Consumer이다.","detected_intent":"예방_정보","confidence":0.93}}

Q: "역류성 식도염을 오래 방치하면 왜 식도암이 생길 수 있나요?"
→ {{"level":"Consumer","reasoning":"'오래 방치하면 왜 ~이 생기나요?' 형식은 일반인의 건강 우려를 표현하는 전형적 문체이다. 질환 합병증에 대한 일반인의 예방 정보 요구이다.","detected_intent":"예방_정보","confidence":0.94}}

Q: "위궤양 환자가 아스피린(aspirin)이나 이부프로펜(ibuprofen)을 피해야 하는 이유는 무엇인가요?"
→ {{"level":"Consumer","reasoning":"약물명이 명시되어 있지만 '피해야 하는 이유'는 환자 본인의 복약 안전을 묻는 일반인 질의이다. 약동학·처방 결정을 묻는 것이 아니라 안전 정보를 구하는 Consumer이다.","detected_intent":"부작용_문의","confidence":0.95}}

Q: "갑상선 호르몬이 부족하면 왜 신진대사가 느려지나요?"
→ {{"level":"Consumer","reasoning":"'호르몬이 부족하면 왜 ~나요?' 형식은 질환 기전을 일상 언어로 묻는 일반인 질의이다. 검사 수치 해석이나 처방 결정과 무관하므로 Consumer이다.","detected_intent":"증상_설명","confidence":0.95}}

Q: "알츠하이머병의 초기 기억력 저하는 나이 들면서 생기는 일반적인 건망증과 어떻게 다른가요?"
→ {{"level":"Consumer","reasoning":"'어떻게 다른가요?' 형식은 일반인이 질환 이해를 위해 묻는 전형적 표현이다. 알츠하이머병이라는 의학 용어가 있지만 일상 언어로 차이를 묻는 Consumer이다.","detected_intent":"증상_설명","confidence":0.94}}

Q: "만성 신장질환이 있으면 왜 빈혈이 생기나요?"
→ {{"level":"Consumer","reasoning":"'왜 ~이 생기나요?' 형식은 일반인의 합병증 원인 궁금증을 표현한다. 만성 신장질환이라는 의학 용어에도 불구하고 일상 언어 패턴이므로 Consumer이다.","detected_intent":"예방_정보","confidence":0.95}}

Q: "본태성 고혈압에서 RAAS의 활성화 기전을 설명하고 ACEI와 ARB의 약리학적 작용 차이를 비교하시오."
→ {{"level":"Professional","reasoning":"RAAS 기전과 약리학적 비교를 요구하는 전문가 지문이다. '비교하시오' 서술형 형식과 약동학 내용이 특징적이다.","detected_intent":"기전_탐구","confidence":0.97}}

Q: "Lung cancer is very common and closely linked to one key risk factor. According to the MSD Manual, lung cancer is the leading cause of cancer death in both men and women. About what percentage of lung cancer cases does the MSD Manual say are linked to cigarette smoking?"
→ {{"level":"Consumer","reasoning":"질병명으로 시작하고 MSD Manual을 인용하지만, 흡연과 폐암의 연관성 비율을 묻는 일반인 건강 정보 질문이다. MSD Manual 인용 자체는 Professional 분류 근거가 아니며 환자 교육 목적이므로 Consumer이다.","detected_intent":"예방_정보","confidence":0.94}}

Q: "People with coronary artery disease sometimes have a fatty plaque inside a coronary artery that can rupture suddenly, which can lead to a heart attack. According to the MSD Manual, what does the ruptured plaque expose that activates platelets and triggers blood clotting inside the artery?"
→ {{"level":"Consumer","reasoning":"환자 관점 서술로 시작하며 일반인이 자신의 질환 기전을 이해하기 위한 질문이다. 기전을 묻더라도 처방 결정이나 약동학과 무관하고 환자 교육 목적이므로 Consumer이다.","detected_intent":"증상_설명","confidence":0.95}}

Q: "When a doctor checks whether someone has asthma, they may use a breathing test called spirometry. What does the MSD Manual say this test measures, and at what point during asthma diagnosis is the test performed relative to giving the person a breathing medication called a beta-adrenergic?"
→ {{"level":"Consumer","reasoning":"검사와 진단 절차를 묻고 있지만 환자가 자신의 진단 과정을 이해하기 위한 질문이다. 처방 결정·약동학이 아닌 환자 교육 목적이므로 Consumer이다.","detected_intent":"예방_정보","confidence":0.94}}

Q: "Hives can appear in response to physical triggers such as cold, heat, exercise, or skin scratching. According to the MSD Manual, how quickly do hives typically appear after exposure to most physical triggers?"
→ {{"level":"Consumer","reasoning":"증상명으로 시작하며 물리적 자극에 대한 반응 속도를 묻는 일반인 질의이다. MSD Manual을 인용하지만 환자가 자신의 증상 패턴을 이해하는 목적이므로 Consumer이다.","detected_intent":"증상_설명","confidence":0.95}}

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
    if _PROFESSIONAL_RULE.search(question):
        return {
            "level": "Professional",
            "confidence": 0.97,
            "reasoning": "규칙 기반: 시험·논술형 지문(기술하시오/설명하시오 등)",
            "detected_intent": "기전_탐구",
        }
    # 규칙 기반 사전 분류: 한국어 일상 문체 패턴 감지 시 LLM 없이 Consumer 확정
    if _CONSUMER_RULE.search(question):
        return {
            "level": "Consumer",
            "confidence": 0.96,
            "reasoning": "규칙 기반: 일상 문체 패턴 감지 (이유는 무엇/왜 ~나요/어떻게 다른가요 등)",
            "detected_intent": "예방_정보",
        }
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
