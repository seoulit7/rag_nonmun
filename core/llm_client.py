from contextvars import ContextVar
from typing import Literal

from langchain_openai import ChatOpenAI

import config.settings as settings

_llm_provider: ContextVar[str] = ContextVar("llm_provider", default="openai")

_LLM_MAX_RETRIES = 6


def set_llm_provider(provider: str) -> object:
    p = (provider or "openai").strip().lower()
    if p not in ("openai", "gemini"):
        p = "openai"
    return _llm_provider.set(p)


def reset_llm_provider(token: object) -> None:
    _llm_provider.reset(token)


def get_llm_provider() -> Literal["openai", "gemini"]:
    return "gemini" if _llm_provider.get() == "gemini" else "openai"


def get_chat_llm(
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> ChatOpenAI:
    """LangChain ChatOpenAI 인스턴스 반환 (OpenAI / Gemini 모두 지원)."""
    if get_llm_provider() == "gemini":
        k = settings.get_gemini_api_key()
        if not k:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        return ChatOpenAI(
            model=model,
            api_key=k,
            base_url=settings.GEMINI_OPENAI_COMPAT_BASE_URL,
            max_retries=_LLM_MAX_RETRIES,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    k = settings.get_openai_api_key()
    if not k:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return ChatOpenAI(
        model=model,
        api_key=k,
        max_retries=_LLM_MAX_RETRIES,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def ragas_async_client():
    """RAGAS 평가용 AsyncAnthropic 클라이언트.

    답변 생성 LLM(OpenAI/Gemini, get_llm_provider() 토글)과 무관하게 항상 Claude를 사용한다.
    같은 모델이 답변 생성과 채점을 겸하면 순환성(circularity) 편향 비판이 생기므로,
    RAGAS 판정 LLM만 별도 provider로 고정 분리한다.
    """
    from anthropic import AsyncAnthropic
    k = settings.get_anthropic_api_key()
    if not k:
        raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
    return AsyncAnthropic(api_key=k, max_retries=_LLM_MAX_RETRIES)


def classifier_model() -> str:
    return settings.GEMINI_AUX_MODEL if get_llm_provider() == "gemini" else settings.CLASSIFIER_LLM_MODEL


def rewriter_model() -> str:
    return settings.GEMINI_AUX_MODEL if get_llm_provider() == "gemini" else settings.TRANSLATE_MODEL


def rag_engine_model() -> str:
    return settings.GEMINI_MODEL if get_llm_provider() == "gemini" else settings.OPENAI_MODEL


def ragas_model() -> str:
    """RAGAS 평가 전용 모델명. 항상 Claude (get_llm_provider() 토글과 무관)."""
    return settings.ANTHROPIC_MODEL
