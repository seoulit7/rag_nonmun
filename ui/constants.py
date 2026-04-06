from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IMG_DIR = _PROJECT_ROOT / "image"

PNG_PATH = next(_IMG_DIR.glob("*.png"), None)
SVG_PATH = next(_IMG_DIR.glob("*.svg"), None)

# Tier 정보 단일 정의 — step_renderers, score_card 등에서 공통 참조
TIER_CONFIGS: Dict[int, Dict[str, str]] = {
    0: {"name": "VectorDB (FAISS)",      "icon": "🗄️", "desc": "MSD 매뉴얼 벡터 검색"},
    1: {"name": "LLM 학습데이터",        "icon": "🧠", "desc": "GPT/Gemini 사전 학습 지식 활용"},
    2: {"name": "웹검색 (DuckDuckGo)",   "icon": "🌐", "desc": "실시간 웹 검색"},
}

SESSION_DEFAULTS: Dict[str, Any] = {
    "logs": [],
    "result": "",
    "detected_level": "",
    "scores": None,
    "search_tier": 0,
    "llm_provider": None,
}
