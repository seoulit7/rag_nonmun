import streamlit as st

from ui.constants import PNG_PATH, SVG_PATH

_PIPELINE_DESCRIPTION = """
---
#### 파이프라인 단계별 설명

**1. level_classifier `Agent`— 사용자 수준 분류**
LLM이 질문 문체·용어를 분석해 `Professional`(의료 전문가) 또는 `Consumer`(일반인)으로 분류합니다.
이후 모든 단계의 쿼리 최적화와 답변 생성 스타일이 이 분류 결과를 따릅니다.

---

**2. adaptive_query_rewriter `Agent` — 쿼리 최적화 / 재작성**
- **최초 실행**: 사용자 질문을 MSD 매뉴얼 검색에 최적화된 영문 쿼리로 변환합니다.
- **재시도/에스컬레이션 시**: 이전 평가 결과(F·AR·CP 점수, 할루시네이션 플래그)를 LLM에 제공해 더 정확한 쿼리로 개선합니다.

---

**3. rag_engine `Agent` — 검색 실행 및 답변 합성**
Tier에 따라 적절한 검색 함수를 직접 호출하고, 검색 결과를 LLM으로 합성하는 에이전트입니다.
도구 선택은 Tier가 결정하며, LLM은 수집된 context를 바탕으로 답변을 생성합니다.

```
rag_engine  ← LangGraph 노드
  ├─ Tier 0 → search_vector_db(query)  직접 호출 (FAISS)        ← Tool
  ├─ Tier 1 → _search_llm_knowledge()  직접 호출 (외부 도구 없음)
  └─ Tier 2 → search_web(query)        직접 호출 (DuckDuckGo)   ← Tool
```

각 도구 호출 결과(context)와 사용자 질문을 조합해 LLM으로 최종 답변을 합성합니다.
(Tier 1은 `_search_llm_knowledge`가 이미 LLM 생성 텍스트를 반환하므로 합성 단계 생략)

---

**4. Tier 0 — VectorDB (FAISS)** · `search_vector_db` `Tool`
MSD 매뉴얼 FAISS 인덱스에서 유사 문서 청크를 의미 검색합니다.

| 평가 후 분기 | 조건 | 동작 |
|---|---|---|
| 성공 | F ≥ 0.8 | output_agent로 이동 |
| 재시도 | F < 0.8, 재시도 < 3회, not critical | query refinement 후 재검색 |
| 즉시 에스컬레이션 | **AR < 0.3** (VectorDB에 관련 내용 없음) | Tier 1으로 즉시 이동 |
| 즉시 에스컬레이션 | **F < 0.3 AND CP < 0.2** (검색 자체가 완전히 빗나감) | Tier 1으로 즉시 이동 |
| 소진 에스컬레이션 | 3회 재시도 소진 후에도 기준 미달 | Tier 1으로 이동 |

> **즉시 에스컬레이션 기준**: AR이 0.3 미만이면 VectorDB에 해당 내용이 없는 것으로 판단합니다.
> 쿼리를 아무리 다듬어도 없는 내용은 찾을 수 없으므로 재시도 없이 바로 Tier 1으로 넘어갑니다.

---

**5. Tier 1 — LLM 학습데이터**
외부 검색 없이 LLM(GPT / Gemini)의 사전 학습 지식을 직접 활용합니다.
1회 시도 후 F ≥ 0.8이면 출력, 기준 미달이면 Tier 2로 에스컬레이션합니다.

---

**6. Tier 2 — 웹검색 (DuckDuckGo)** · `search_web` `Tool`
인터넷에서 최신 의료 정보를 수집합니다.
검색된 context와 사용자 질문을 프롬프트로 조합해 LLM으로 답변을 합성합니다.
1회 시도 후 F ≥ 0.8이면 출력, 기준 미달이면 fallback으로 이동합니다.

---

**7. critic_agent `Agent` — RAGAS 품질 평가**
LLM 기반 RAGAS 평가기로 3가지 지표를 산출합니다.

| 지표 | 의미 | 기준값 |
|---|---|---|
| **Faithfulness (F)** | 답변이 검색된 context에 근거하는가 (사실성) | ≥ 0.8 통과 |
| **Answer Relevance (AR)** | 답변이 질문에 얼마나 관련 있는가 | < 0.3 즉시 에스컬레이션 |
| **Context Precision (CP)** | 검색 청크가 얼마나 정밀하게 관련 있는가 | F<0.3 & CP<0.2 즉시 에스컬레이션 |

---

**8. output_agent `Agent` — 최종 출력**
LLM이 영문 답변을 한국어로 번역하고, 검색 출처(MSD 매뉴얼 파일·페이지 / LLM / 웹 URL)와 면책 조항을 추가합니다.

---

**9. fallback — 원문 제시**
모든 Tier를 소진하고도 F < 0.8이면 신뢰할 수 있는 답변을 생성하지 못한 것으로 판단하고,
검색된 원문 자료를 그대로 제시합니다.
"""


def render_header() -> None:
    """메인 타이틀, 파이프라인 expander, 다운로드 버튼을 렌더링한다."""
    st.title("Medical Self-Corrective RAG")
    st.markdown(
        "질문 → 수준 분류 → 쿼리 최적화 → 검색(VectorDB→LLM→웹) → RAGAS 평가 → 자가 교정 → 한국어 번역 답변"
    )

    with st.expander("LangGraph 파이프라인 시각화 및 설명"):
        if PNG_PATH:
            st.image(str(PNG_PATH), width="stretch")
        else:
            st.warning("image/ 폴더에 PNG 파일이 없습니다.")
        st.markdown(_PIPELINE_DESCRIPTION)

    col_png, col_svg, _ = st.columns([1, 1, 4])
    if PNG_PATH:
        col_png.download_button(
            label="⬇ PNG 다운로드",
            data=PNG_PATH.read_bytes(),
            file_name=PNG_PATH.name,
            mime="image/png",
        )
    if SVG_PATH:
        col_svg.download_button(
            label="⬇ SVG 다운로드",
            data=SVG_PATH.read_bytes(),
            file_name=SVG_PATH.name,
            mime="image/svg+xml",
        )
