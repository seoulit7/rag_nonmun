import logging
import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_k):
        return False

_ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 루트
_ENV_FILE = _ROOT / ".env"
load_dotenv(_ENV_FILE)
if not (os.environ.get("OPENAI_API_KEY") or "").strip():
    load_dotenv(_ENV_FILE, override=True)

# Streamlit Cloud Secrets → os.environ 주입
# st.secrets는 모듈 import 시점에도 접근 가능하며,
# os.environ에 없는 키만 주입하여 로컬 .env 값을 덮어쓰지 않는다.
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass


def _env(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    return (v if v is not None else default).strip()


def _parsed_openai_api_key_from_file(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, _, val = line.partition("=")
        if name.strip() != "OPENAI_API_KEY":
            continue
        v = val.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        return v.strip()
    return ""


def get_gemini_api_key() -> str:
    return _env("GEMINI_API_KEY", "")


def get_openai_api_key() -> str:
    k = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if k:
        return k
    load_dotenv(_ENV_FILE, override=True)
    k = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if k:
        return k
    k = _parsed_openai_api_key_from_file(_ENV_FILE)
    if k:
        os.environ["OPENAI_API_KEY"] = k
        return k
    return ""


def resolve_project_path(rel_or_abs: str, default_rel: str) -> str:
    raw = (rel_or_abs or "").strip() or default_rel
    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())
    return str((_ROOT / raw).resolve())


DATA_DIR = resolve_project_path(_env("MEDICAL_RAG_DATA_DIR", ""), "data")
INDEX_PATH = resolve_project_path(_env("MEDICAL_RAG_INDEX_PATH", ""), "db/msd_faiss.index")
EMBEDDING_MODEL = _env(
    "MEDICAL_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LOG_LEVEL = _env("MEDICAL_RAG_LOG_LEVEL", "INFO").upper()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


CHUNK_MAX_CHARS = _env_int("MEDICAL_RAG_CHUNK_MAX_CHARS", 500)
CHUNK_OVERLAP = _env_int("MEDICAL_RAG_CHUNK_OVERLAP", 60)
RAG_TOP_K = max(1, _env_int("MEDICAL_RAG_TOP_K", 2))

OPENAI_API_KEY = get_openai_api_key()
GEMINI_OPENAI_COMPAT_BASE_URL = _env(
    "GEMINI_OPENAI_COMPAT_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-pro")
_gemini_aux = _env("MEDICAL_RAG_GEMINI_AUX_MODEL", "")
GEMINI_AUX_MODEL = _gemini_aux if _gemini_aux else "gemini-2.5-flash"
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o")
TRANSLATE_MODEL = _env("MEDICAL_RAG_TRANSLATE_MODEL", "gpt-4o-mini")
RAGAS_LLM_MODEL = _env("MEDICAL_RAG_RAGAS_LLM_MODEL", "gpt-4o-mini")
CLASSIFIER_LLM_MODEL = _env("MEDICAL_RAG_CLASSIFIER_MODEL", "gpt-4o-mini")

# Self-Correction Loop 제어
MAX_LOOPS = _env_int("MEDICAL_RAG_MAX_LOOPS", 3)
FAITHFULNESS_THRESHOLD = float(_env("MEDICAL_RAG_FAITHFULNESS_THRESHOLD", "0.8"))
AR_THRESHOLD = float(_env("MEDICAL_RAG_AR_THRESHOLD", "0.8"))
CP_THRESHOLD = float(_env("MEDICAL_RAG_CP_THRESHOLD", "0.8"))

# Tier 0 즉시 에스컬레이션 임계값 (query rewriting 없이 바로 다음 Tier로)
# AR < CRITICAL_AR: VectorDB에 관련 내용 자체가 없다고 판단
CRITICAL_AR_THRESHOLD = float(_env("MEDICAL_RAG_CRITICAL_AR_THRESHOLD", "0.3"))
# F < CRITICAL_F AND CP < CRITICAL_CP: 검색 자체가 완전히 빗나간 경우
CRITICAL_F_THRESHOLD = float(_env("MEDICAL_RAG_CRITICAL_F_THRESHOLD", "0.3"))
CRITICAL_CP_THRESHOLD = float(_env("MEDICAL_RAG_CRITICAL_CP_THRESHOLD", "0.2"))

# 웹 검색 파라미터
WEB_SEARCH_MAX_RESULTS = _env_int("MEDICAL_RAG_WEB_SEARCH_MAX_RESULTS", 3)

# RAGAS 평가 파라미터
RAGAS_STRICTNESS = _env_int("MEDICAL_RAG_RAGAS_STRICTNESS", 3)
RAGAS_ANSWER_MAX_CHARS = _env_int("MEDICAL_RAG_RAGAS_ANSWER_MAX_CHARS", 1500)
RAGAS_CONTEXT_MAX_CHARS = _env_int("MEDICAL_RAG_RAGAS_CONTEXT_MAX_CHARS", 2000)
# Faithfulness JSON 생성 시 출력 토큰 한도 (기본 3072는 긴 의료 텍스트에서 부족)
RAGAS_LLM_MAX_TOKENS = _env_int("MEDICAL_RAG_RAGAS_LLM_MAX_TOKENS", 8192)

# Supabase audit log
SUPABASE_DB_URL = _env("SUPABASE_DB_URL", "")

# PDF OCR (스캔 PDF 텍스트 추출)
# rapidocr-onnxruntime 패키지 필요. true로 설정하면 PyPDFLoader extract_images=True 사용.
PDF_OCR_ENABLED = _env("MEDICAL_RAG_PDF_OCR", "false").lower() in ("1", "true", "yes")

if _env("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if _env("LANGSMITH_ENDPOINT"):
        os.environ.setdefault("LANGCHAIN_ENDPOINT", _env("LANGSMITH_ENDPOINT"))
    if _env("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", _env("LANGSMITH_API_KEY"))
    if _env("LANGSMITH_PROJECT"):
        os.environ.setdefault("LANGCHAIN_PROJECT", _env("LANGSMITH_PROJECT"))


def setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(level=level)


def ensure_tesseract_on_path() -> None:
    exe = os.environ.get("MEDICAL_RAG_TESSERACT_EXE", "").strip()
    if exe and os.path.isfile(exe):
        d = os.path.dirname(os.path.normpath(exe))
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        td = os.path.join(d, "tessdata")
        if os.path.isdir(td):
            os.environ["TESSDATA_PREFIX"] = td
        return
    if shutil.which("tesseract"):
        wh = shutil.which("tesseract")
        if wh:
            d = os.path.dirname(os.path.normpath(wh))
            td = os.path.join(d, "tessdata")
            if os.path.isdir(td):
                os.environ["TESSDATA_PREFIX"] = td
        return
    for root in (
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
    ):
        texe = os.path.join(root, "tesseract.exe")
        if os.path.isfile(texe):
            d = os.path.dirname(texe)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            td = os.path.join(root, "tessdata")
            if os.path.isdir(td):
                os.environ["TESSDATA_PREFIX"] = td
            break
