from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IMG_DIR = _PROJECT_ROOT / "image"

PNG_PATH = _IMG_DIR / "pipeline_diagram.png"
SVG_PATH = _IMG_DIR / "pipeline_diagram.svg"

# Tier 정보 단일 정의 — step_renderers, score_card 등에서 공통 참조
TIER_CONFIGS: Dict[int, Dict[str, str]] = {
    0: {"name": "VectorDB (FAISS)",        "icon": "🗄️", "desc": "MSD Manual vector search"},
    1: {"name": "LLM Knowledge",           "icon": "🧠", "desc": "GPT/Gemini pre-trained knowledge"},
    2: {"name": "Web Search (DuckDuckGo)", "icon": "🌐", "desc": "Real-time web search"},
}

SESSION_DEFAULTS: Dict[str, Any] = {
    "logs": [],
    "result": "",
    "detected_level": "",
    "scores": None,
    "search_tier": 0,
    "llm_provider": None,
}
