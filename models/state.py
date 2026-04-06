from typing import TypedDict, List


class GraphState(TypedDict):
    question: str
    user_level: str
    queries: List[str]
    context: List[str]
    context_sources: List[str]        # 검색된 청크의 원본 메타데이터 (파일명#페이지 or 소스명)
    answer: str
    critic_score: float               # Faithfulness (주 게이트, 0–1)
    answer_relevance_score: float     # Answer Relevance (역 질문 기반)
    context_precision_score: float    # Context Precision (청크 유효성)
    hallucination_flags: List[str]    # 할루시네이션·미지지 주장 목록
    search_tier: int                  # 0=VectorDB, 1=LLM 학습데이터, 2=웹검색
    llm_provider: str
    loop_count: int
    log: List[str]


TIER_LABELS = {0: "VectorDB(FAISS)", 1: "LLM 학습데이터", 2: "웹검색(DuckDuckGo)"}
