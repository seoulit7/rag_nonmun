# Medical Self-Corrective RAG 파이프라인

환경 변수 기본값은 `config/settings.py`·`.env`와 동일하게 맞춘다.

## 1. level_classifier Agent

사용자 수준 분류 LLM이 질문 문체·용어를 분석해 Professional(의료 전문가) 또는 Consumer(일반인)으로 분류한다. 이후 모든 단계의 쿼리 최적화와 답변 생성 스타일이 이 분류 결과를 따른다.

## 2. adaptive_query_rewriter Agent — 쿼리 최적화 / 재작성

- **최초 실행:** 사용자 질문을 MSD 매뉴얼 검색에 최적화된 영문 쿼리로 변환한다.
- **재시도/에스컬레이션 시:** 이전 평가 결과(F·**AR**·CP 점수, 할루시네이션 플래그)를 LLM에 제공해 더 정확한 쿼리로 개선한다.

## 3. rag_engine Agent — 검색 실행 및 답변 합성

Tier에 따라 적절한 검색 함수를 직접 호출하고, 검색 결과를 LLM으로 합성한다. 도구 선택은 Tier가 결정하며, LLM은 수집된 context를 바탕으로 답변을 생성한다.

```
rag_engine  ← LangGraph 노드
  ├─ Tier 0 → search_vector_db(query)  직접 호출 (FAISS)        ← Tool
  ├─ Tier 1 → _search_llm_knowledge()  직접 호출 (외부 도구 없음)
  └─ Tier 2 → search_web(query)        직접 호출 (DuckDuckGo)   ← Tool
```

각 도구 호출 결과(context)와 사용자 질문을 조합해 LLM으로 최종 답변을 합성한다. (Tier 1은 `_search_llm_knowledge`가 이미 LLM 생성 텍스트를 반환하므로 합성 단계 생략)

## 4. Tier 0 — VectorDB (FAISS) · search_vector_db Tool

MSD 매뉴얼 FAISS 인덱스에서 유사 문서 청크를 의미 검색한다.

_실행 순서(한 사이클 평가 후):_

| 분기 | 조건 | 동작 |
|------|------|------|
| 즉시 에스컬레이션 | `AR < MEDICAL_RAG_CRITICAL_AR_THRESHOLD` (기본 0.3) | Tier 1으로 즉시 이동 |
| 즉시 에스컬레이션 | `F < MEDICAL_RAG_CRITICAL_F_THRESHOLD` **且** `CP < MEDICAL_RAG_CRITICAL_CP_THRESHOLD` (기본 0.3 / 0.2) | Tier 1으로 즉시 이동 |
| 성공 | `F ≥ MEDICAL_RAG_FAITHFULNESS_THRESHOLD` (기본 0.8) | output_agent로 이동 |
| 재시도 | 위에 해당 없고 재시도 횟수가 `MEDICAL_RAG_MAX_LOOPS` 미만 | query refinement 후 재검색 |
| 소진 에스컬레이션 | `MEDICAL_RAG_MAX_LOOPS`회 재시도 소진 후에도 기준 미달 | Tier 1으로 이동 |

**즉시 에스컬레이션:** `AR`이 `MEDICAL_RAG_CRITICAL_AR_THRESHOLD` 미만이면 VectorDB에 해당 내용이 없다고 판단한다. 쿼리를 아무리 다듬어도 없는 내용은 찾기 어려우므로, **Faithfulness가 높아도** 재시도 없이 바로 Tier 1으로 넘어간다.

## 5. Tier 1 — LLM 학습데이터

외부 검색 없이 LLM(GPT / Gemini)의 사전 학습 지식을 직접 활용한다. 1회 시도 후 `F ≥ FAITHFULNESS_THRESHOLD`이면 출력, 기준 미달이면 Tier 2로 에스컬레이션한다.

## 6. Tier 2 — 웹검색 (DuckDuckGo) · search_web Tool

인터넷에서 최신 의료 정보를 수집한다. 검색된 context와 사용자 질문을 프롬프트로 조합해 LLM으로 답변을 합성한다. 1회 시도 후 `F ≥ FAITHFULNESS_THRESHOLD`이면 출력, 기준 미달이면 fallback으로 이동한다.

## 7. critic_agent Agent — RAGAS 품질 평가

LLM 기반 RAGAS 평가기로 3가지 지표를 산출한다.

| 지표 | 의미 | 코드·환경 변수와의 대응 |
|------|------|-------------------------|
| Faithfulness (F) | 답변이 검색된 context에 근거하는가 | `MEDICAL_RAG_FAITHFULNESS_THRESHOLD` (기본 0.8) |
| Answer Relevance (AR) | 답변이 질문에 얼마나 관련 있는가 | Tier 0 즉시 에스컬: `AR < MEDICAL_RAG_CRITICAL_AR_THRESHOLD` |
| Context Precision (CP) | 검색 청크가 질의에 얼마나 정밀한가 | 재쿼리 플래그: `CP < MEDICAL_RAG_CONTEXT_REQUERY_THRESHOLD`; F·CP 동시 저조 시 즉시 에스컬은 `CRITICAL_F`·`CRITICAL_CP` |

## 8. output_agent Agent — 최종 출력

LLM이 답변 본문을 한국어로 번역하고(원문이 영·중·일 등일 수 있음), 검색 출처(MSD 매뉴얼 파일·페이지 / LLM / 웹 URL)와 면책 조항을 추가한다.

## 9. fallback — 원문 제시

모든 Tier를 소진하고도 `F < FAITHFULNESS_THRESHOLD`이면 신뢰할 수 있는 답변을 생성하지 못한 것으로 판단하고, 검색된 원문 자료를 그대로 제시한다.
