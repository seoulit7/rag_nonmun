import json
import re


def parse_llm_json(content: str) -> dict:
    """LLM이 반환한 문자열에서 JSON 객체를 추출한다."""
    raw = (content or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, count=1, flags=re.IGNORECASE)
        end_fence = raw.rfind("```")
        if end_fence >= 0:
            raw = raw[:end_fence].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    i0, i1 = raw.find("{"), raw.rfind("}")
    if i0 >= 0 and i1 > i0:
        frag = raw[i0: i1 + 1]
        try:
            return json.loads(frag)
        except json.JSONDecodeError:
            pass
    return {}


def fallback_optimizer_json(text: str) -> dict:
    """query/reasoning 필드를 정규식으로 추출하는 폴백 파서."""
    qm = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    rm = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    d: dict = {}
    if qm:
        try:
            d["query"] = json.loads('"' + qm.group(1).replace("\n", " ") + '"')
        except json.JSONDecodeError:
            d["query"] = qm.group(1).replace("\\n", " ").strip()
    if rm:
        try:
            d["reasoning"] = json.loads('"' + rm.group(1).replace("\n", " ") + '"')
        except json.JSONDecodeError:
            d["reasoning"] = rm.group(1).replace("\\n", " ").strip()
    return d


def fallback_classifier_json(text: str) -> dict:
    """level/confidence/intent/reasoning 필드를 정규식으로 추출하는 폴백 파서."""
    lm = re.search(r'"level"\s*:\s*"((?:Professional|Consumer))"', text, re.I)
    cm = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    im = re.search(r'"detected_intent"\s*:\s*"([^"]*)"', text)
    rm = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    d: dict = {}
    if lm:
        d["level"] = (
            "Professional" if lm.group(1).lower() == "professional" else "Consumer"
        )
    if cm:
        d["confidence"] = float(cm.group(1))
    if im:
        d["detected_intent"] = im.group(1)
    if rm:
        try:
            d["reasoning"] = json.loads('"' + rm.group(1) + '"')
        except json.JSONDecodeError:
            d["reasoning"] = rm.group(1).strip()
    return d
