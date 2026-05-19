import asyncio
import logging
import math
import re
from typing import List, NamedTuple, Sequence

import config.settings as settings
from core.llm_client import ragas_async_client, ragas_model
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_precision import ContextPrecisionWithoutReference
from ragas.metrics.collections.faithfulness import Faithfulness

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Flesch-Kincaid Grade Level (영어 원문 기준)
# 공식: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
# 번역 전 영어 RAG 답변에 대해 계산한다.
# Consumer 목표: ≤ 9  /  Professional 기준: ≥ 12
# ──────────────────────────────────────────────────────────────────────────────
def _syllables_en(word: str) -> int:
    """영어 단어의 음절 수를 근사 계산한다."""
    import re
    word = word.lower().rstrip(".,!?;:'\"")
    if not word:
        return 0
    word = re.sub(r"e$", "", word)          # 묵음 e 제거
    count = len(re.findall(r"[aeiou]+", word))
    return max(1, count)


def flesch_kincaid_grade_en(text: str) -> float:
    """영어 텍스트의 Flesch-Kincaid Grade Level을 계산한다.

    번역 전 영어 원문(state['answer'])을 인자로 받아야 한다.
    """
    import re
    if not text or not text.strip():
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    words     = re.findall(r"\b[a-zA-Z]+\b", text)

    n_sent = max(len(sentences), 1)
    n_word = max(len(words), 1)
    n_syl  = sum(_syllables_en(w) for w in words)

    grade = 0.39 * (n_word / n_sent) + 11.8 * (n_syl / n_word) - 15.59
    return round(max(0.0, grade), 3)


class OfficialRagasScores(NamedTuple):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    hallu_flags: List[str]


# ──────────────────────────────────────────────────────────────────────────────
# 의료 도메인 할루시네이션 탐지 패턴
# ──────────────────────────────────────────────────────────────────────────────
_HALLU_PATTERNS = [
    (re.compile(r"\d+(?:\.\d+)?\s*(?:mg|ml|mcg|μg|g|L|%|회|정|캡슐)"), "수치"),
    (re.compile(r"[가-힣A-Za-z]{2,}(?:\s*\+\s*[가-힣A-Za-z]{2,})+"), "약물 배합"),
    (re.compile(r"(?:1|2|3|4|5)(?:단계|차\s*치료|선\s*치료|라인)"), "치료 단계"),
]


def _detect_hallu_flags(answer: str, contexts: List[str]) -> List[str]:
    ctx = " ".join(contexts)
    flags: List[str] = []
    for pattern, label in _HALLU_PATTERNS:
        ans_matches = set(pattern.findall(answer))
        ctx_matches = set(pattern.findall(ctx))
        for m in ans_matches - ctx_matches:
            flags.append(f"[Hallucination:{label}] '{m}'")
    return flags


def _safe_unit(v: float) -> float:
    if v is None:
        return 0.0
    x = float(v)
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _prep_contexts(chunks: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in chunks:
        t = (c or "").strip()
        if t:
            out.append(t[:settings.RAGAS_CONTEXT_MAX_CHARS])
    if not out:
        out = ["(검색된 컨텍스트 없음)"]
    return out


def compute_official_ragas_scores(
    question: str,
    answer_body: str,
    context_chunks: Sequence[str],
) -> OfficialRagasScores:
    q = (question or "").strip()[:500]
    a = (answer_body or "").strip()[:settings.RAGAS_ANSWER_MAX_CHARS]
    ctx_list = _prep_contexts(context_chunks)

    eval_client = ragas_async_client()
    llm = llm_factory(
        ragas_model(),
        client=eval_client,
        temperature=0,
        max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
    )
    embedder = HuggingFaceEmbeddings(model=settings.EMBEDDING_MODEL)

    faith = Faithfulness(llm=llm)
    arel = AnswerRelevancy(llm=llm, embeddings=embedder, strictness=settings.RAGAS_STRICTNESS)
    cpre = ContextPrecisionWithoutReference(llm=llm)

    async def _run_all() -> OfficialRagasScores:
        async def _score_faith():
            try:
                r = await faith.ascore(user_input=q, response=a, retrieved_contexts=ctx_list)
                return _safe_unit(r.value)
            except Exception as e:
                logger.warning("Faithfulness 평가 실패: %s", e, exc_info=True)
                return 0.0

        async def _score_arel():
            try:
                r = await arel.ascore(user_input=q, response=a)
                return _safe_unit(r.value)
            except Exception as e:
                logger.warning("AnswerRelevancy 평가 실패: %s", e, exc_info=True)
                return 0.0

        async def _score_cpre():
            try:
                r = await cpre.ascore(user_input=q, response=a, retrieved_contexts=ctx_list)
                return _safe_unit(r.value)
            except Exception as e:
                logger.warning("ContextPrecision 평가 실패: %s", e, exc_info=True)
                return 0.0

        try:
            ff, ar, cp = await asyncio.gather(_score_faith(), _score_arel(), _score_cpre())
        finally:
            # httpx AsyncClient를 루프 종료 전에 명시적으로 닫아
            # "Event loop is closed" 경고를 방지한다.
            try:
                await eval_client.aclose()
            except Exception:
                pass

        logger.warning("[RAGAS] scores F=%.3f AR=%.3f CP=%.3f | q=%r | a=%r | ctx=%d",
                       ff, ar, cp, q[:60], a[:60], len(ctx_list))

        hallu_flags = _detect_hallu_flags(a, ctx_list)
        return OfficialRagasScores(
            faithfulness=ff,
            answer_relevance=ar,
            context_precision=cp,
            hallu_flags=hallu_flags,
        )

    # Streamlit은 자체 이벤트 루프를 보유하므로 asyncio.run() 직접 호출 시
    # "This event loop is already running" 오류가 발생한다.
    # 별도 스레드에서 새 이벤트 루프를 생성해 실행하면 충돌을 피할 수 있다.
    import concurrent.futures

    def _run_in_thread() -> OfficialRagasScores:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(_run_all())
        finally:
            # pending 태스크를 모두 정리한 후 루프를 닫아
            # "Event loop is closed" 경고를 방지한다.
            try:
                pending = asyncio.all_tasks(new_loop)
                if pending:
                    new_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                new_loop.run_until_complete(new_loop.shutdown_asyncgens())
            except Exception:
                pass
            new_loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run_in_thread)
            return future.result(timeout=120)
    except Exception as e:
        logger.error("RAGAS 평가 전체 실패: %s", e, exc_info=True)
        return OfficialRagasScores(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_precision=0.0,
            hallu_flags=[],
        )
