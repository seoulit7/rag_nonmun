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
# Consumer 목표: ≤ 10  /  Professional 기준: ≥ 12
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


# ── Consumer FK Grade 의학 용어 마스킹 (음절 왜곡 제거) ───────────────────────
# 패턴 1: 그리스/라틴 의학 접미사 (-itis, -ology, -osis, -oma 등)
_PURE_FK_SUFFIX = re.compile(
    r"\b\w+(?:itis|ology|ectomy|plasty|oscopy|otomy|ostomy|algia|pathy"
    r"|emia|uria|graphy|trophy|genesis|lysis|osis|oma"
    r"|pnea|plasia|ectasis|rrhea)\b",
    re.IGNORECASE,
)
# 패턴 2: 소유격 병명 (Alzheimer's disease, Parkinson's syndrome 등)
_PURE_FK_EPONYM = re.compile(r"\b[A-Z][a-z]+'s\s+(?:disease|syndrome|disorder)\b")
# 패턴 3: 10자 이상 단어 (cardiovascular, bronchoconstriction, medication 등)
_PURE_FK_LONG   = re.compile(r"\b[A-Za-z]{10,}\b")
# 패턴 4: FAISS 원문 빈도 기반 명시 목록 (6-9자, 접미사·길이 패턴 미탐지)
_PURE_FK_EXPLICIT = frozenset({
    # Cardiovascular
    "hypertension", "coronary",    "myocardial",  "infarction",
    "arrhythmia",   "cholesterol", "angiotensin", "ischemic",
    "vascular",     "diastolic",   "cerebral",    "angina",
    # Respiratory
    "pulmonary",    "obstructive", "respiratory", "pneumonia",
    "bronchospasm", "emphysema",   "influenza",   "pleurisy",
    "pleuritic",    "sputum",      "exudate",     "atypical",
    # GI / Digestive
    "esophagus",    "stricture",   "pyloric",     "duodenal",
    "peptic",       "gastric",     "duodenum",    "mucosal",
    "mucosa",       "reflux",
    # Metabolic / Endocrine
    "diabetes",     "mellitus",    "metabolic",   "endocrine",
    "deficiency",   "potassium",   "estrogen",    "progesterone",
    "creatinine",   "metformin",   "thyroid",     "glycemic",
    # Oncology
    "carcinoma",    "colorectal",  "chemotherapy","tamoxifen",
    "metastatic",   "lymphoma",    "sarcoma",     "melanoma",
    "rectal",       "sigmoid",     "polyp",       "biopsy",
    # Neurology / Psychiatry
    "dementia",     "alzheimer",   "depression",  "depressive",
    "neurologic",   "cognitive",   "serotonin",   "dopamine",
    "confusion",
    # Musculoskeletal
    "synovial",     "cartilage",
    # Immunology / Inflammation
    "inflammatory", "inflammation","bacterial",   "antibiotics",
    "antibiotic",   "neutrophil",  "hormonal",    "antiviral",
    # Infectious disease / COVID
    "variant",      "quarantine",
    # General clinical symptoms
    "malaise",      "vomiting",
    # General medical
    "receptor",     "receptors",   "inhibitor",   "diagnostic",
    "impairment",   "mortality",   "hemorrhage",  "abdominal",
    # 6-9자 고빈도 의학 용어 (3음절 이상, 패턴 1·3 미탐지)
    "nausea",       "oxygen",
    "therapy",      "surgery",     "cardiac",     "calcium",
    "insulin",      "vitamin",     "hepatic",     "urinary",
    "aspirin",      "medicine",    "bacteria",
    "molecule",     "antibody",    "nutrient",    "cellular",
    "clinical",     "physical",    "surgical",
    "infection",    "condition",   "procedure",   "physician",
    "ibuprofen",
    # Psychiatric / mood (우울증·불안장애 답변)
    "insomnia",     "anhedonia",   "dysthymia",
    # Anxiety / autonomic (불안장애 답변)
    "dizziness",    "trembling",
    # CKD (만성신장질환 답변)
    "nocturia",
    # Breast / oncology (유방암 답변)
    "mammogram",    "palpable",
    # General clinical (갑상선·빈혈·천식 답변)
    "fatigue",      "prognosis",   "inhaler",
    # Vascular / cardiovascular (고혈압·뇌졸중 답변)
    "arteries",     "arterial",    "venous",      "venules",
    "narrowing",    "narrowed",    "deposits",    "clotting",
    "elevated",     "ruptures",    "stenotic",
    # Hematology (빈혈 답변)
    "erythrocyte",  "ferritin",    "serum",       "marrow",
    "carries",      "carrying",    "endurance",
    # Immune / infection (폐렴·불안 답변)
    "triggers",     "mediates",    "activates",   "cascade",
    "pathogens",    "pathogen",    "microbes",
})
_PURE_FK_EXPLICIT_PAT = re.compile(
    r"\b(?:" + "|".join(sorted(_PURE_FK_EXPLICIT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def get_pure_fk_grade(text: str) -> float:
    """의학 전문 용어를 'it'으로 마스킹한 순수 문장 구조의 FK Grade를 반환한다.

    Consumer 답변의 실제 가독성을 측정한다. (의학 용어 음절 수 왜곡 제거)
    4단계 마스킹: 접미사+oma → 소유격 병명 → 10자 이상 → FAISS 빈도 기반 명시 목록
    """
    masked = _PURE_FK_SUFFIX.sub("it", text)
    masked = _PURE_FK_EPONYM.sub("it", masked)
    masked = _PURE_FK_LONG.sub("it", masked)
    masked = _PURE_FK_EXPLICIT_PAT.sub("it", masked)
    return flesch_kincaid_grade_en(masked)


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
    ar_query: str = None,
) -> OfficialRagasScores:
    q = (question or "").strip()[:500]
    # AR은 질문 의도가 고정된 첫 번째 쿼리로 평가 (재시도마다 쿼리가 바뀌면 AR이 폭락)
    q_ar = (ar_query or question or "").strip()[:500]
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
                r = await arel.ascore(user_input=q_ar, response=a)
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

        logger.warning("[RAGAS] scores F=%.3f AR=%.3f CP=%.3f | q=%r | ar_q=%r | a=%r | ctx=%d",
                       ff, ar, cp, q[:60], q_ar[:60], a[:60], len(ctx_list))

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
