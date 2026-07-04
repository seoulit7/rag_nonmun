# Medical Self-Corrective RAG 파이프라인

환경 변수 기본값은 `config/settings.py`·`.env`와 동일하게 맞춘다.

## 1. level_classifier Agent

사용자 수준 분류 LLM이 질문 문체·용어를 분석해 Professional(의료 전문가) 또는 Consumer(일반인)으로 분류한다. 이후 모든 단계의 쿼리 최적화와 답변 생성 스타일이 이 분류 결과를 따른다.

> **조건 E (Baseline)**: `run_medical_self_corrective_rag()` 호출 시 `forced_user_level="Baseline"`으로 강제 고정되어 이 노드의 LLM 분류를 우회한다.

## 2. adaptive_query_rewriter Agent — 쿼리 최적화 / 재작성

- **최초 실행:** 사용자 질문을 MSD 매뉴얼 검색에 최적화된 영문 쿼리로 변환한다.
- **재시도/에스컬레이션 시:** 이전 평가 결과(F·**AR**·CP 점수, 할루시네이션 플래그, critic_feedback)를 LLM에 제공해 더 정확한 쿼리로 개선한다.

## 3. rag_engine Agent — 검색 실행 및 답변 합성

Tier에 따라 적절한 검색 함수를 직접 호출하고, 검색 결과를 LLM으로 합성한다. 도구 선택은 Tier가 결정하며, LLM은 수집된 context를 바탕으로 답변을 생성한다.

```
rag_engine  ← LangGraph 노드
  ├─ Tier 0 → search_msd_manual(query)  ReAct 에이전트 (FAISS)       ← Tool
  ├─ Tier 1 → _search_llm_knowledge()   직접 호출 (외부 도구 없음)
  └─ Tier 2 → search_web(query)         ReAct 에이전트 (DuckDuckGo)  ← Tool
```

각 도구 호출 결과(context)와 사용자 질문을 조합해 LLM으로 최종 답변을 합성한다. (Tier 1은 `_search_llm_knowledge`가 이미 LLM 생성 텍스트를 반환하므로 합성 단계 생략)

**사용자 수준별 출력 스타일 (Flesch-Kincaid Grade Level 목표):**

| 수준 | FK Grade 목표 | 언어 규칙 |
|------|--------------|-----------|
| Consumer | ≤ 9 | 15단어 이하 문장, 1~2음절 일상어, 의료 용어 시 괄호 설명 |
| Professional | ≥ 12 | 20단어 이상 복합 문장, 임상·약리 전문 용어, 라틴/그리스어 어근 미설명 |

## 4. Tier 0 — VectorDB (FAISS) · search_msd_manual Tool

MSD 매뉴얼 FAISS 인덱스에서 유사 문서 청크를 코사인 유사도로 검색한다.  
임베딩 모델: `BAAI/bge-base-en-v1.5` (768차원)

_실행 순서(한 사이클 평가 후):_

| 분기 | 조건 | 동작 |
|------|------|------|
| 즉시 에스컬레이션 | `AR < CRITICAL_AR_THRESHOLD` (기본 0.3) | Tier 1으로 즉시 이동 |
| 즉시 에스컬레이션 | `F < CRITICAL_F_THRESHOLD` **且** `CP < CRITICAL_CP_THRESHOLD` (기본 0.3 / 0.2) | Tier 1으로 즉시 이동 |
| 성공 | `F ≥ 0.8 AND AR ≥ 0.8 AND CP ≥ 0.8` | output_agent로 이동 |
| 재시도 | 위에 해당 없고 재시도 횟수가 `MAX_LOOPS` 미만 | query refinement 후 재검색 |
| 소진 에스컬레이션 | `MAX_LOOPS`회 재시도 소진 후에도 기준 미달 | Tier 1으로 이동 |

**즉시 에스컬레이션:** `AR`이 `CRITICAL_AR_THRESHOLD` 미만이면 VectorDB에 해당 내용이 없다고 판단한다. 쿼리를 아무리 다듬어도 없는 내용은 찾기 어려우므로, **Faithfulness가 높아도** 재시도 없이 바로 Tier 1으로 넘어간다.

## 5. Tier 1 — LLM 학습데이터

외부 검색 없이 LLM(GPT / Gemini)의 사전 학습 지식을 직접 활용한다. 1회 시도 후 **AR ≥ AR_THRESHOLD(0.8)** 이면 출력, 기준 미달이면 Tier 2로 에스컬레이션한다.

> **평가 지표**: Tier 1은 AR(Answer Relevance)만 평가한다. LLM 학습데이터 기반 답변은 컨텍스트 청크가 없으므로 F·CP를 적용하지 않는다. 중간 루프 로그(save_loop_log)에도 F·CP는 NULL로 저장된다.

## 6. Tier 2 — 웹검색 (DuckDuckGo) · search_web Tool

인터넷에서 최신 의료 정보를 수집한다. 검색된 context와 사용자 질문을 프롬프트로 조합해 LLM으로 답변을 합성한다. 1회 시도 후 `F ≥ 0.8 AND AR ≥ 0.8 AND CP ≥ 0.8` 이면 출력, 기준 미달이면 fallback으로 이동한다.

## 7. critic_agent Agent — RAGAS 품질 평가

LLM 기반 RAGAS 평가기로 3가지 지표를 산출한다.

| 지표 | 의미 | 코드·환경 변수와의 대응 |
|------|------|-------------------------|
| Faithfulness (F) | 답변이 검색된 context에 근거하는가 | `FAITHFULNESS_THRESHOLD` (기본 0.8) |
| Answer Relevance (AR) | 답변이 질문에 얼마나 관련 있는가 | Tier 0 즉시 에스컬: `AR < CRITICAL_AR_THRESHOLD`; Tier 1 단독 평가 지표 |
| Context Precision (CP) | 검색 청크가 질의에 얼마나 정밀한가 | F·CP 동시 저조 시 즉시 에스컬 → `CRITICAL_F`·`CRITICAL_CP` |

평가 완료 후 `critic_feedback` (개선 힌트 — 기준 미달 지표와 원인을 자연어로 요약한 문자열)을 생성한다.

> **AR 첫 쿼리 고정**: AR(Answer Relevance)은 재시도·에스컬레이션으로 쿼리가 바뀌어도 **항상 첫 번째 쿼리(`queries[0]`)를 기준**으로 평가한다. 쿼리를 변경할수록 원래 질문 의도와 멀어져 AR 기준이 흔들리는 문제를 방지하기 위한 설계다.

**루프 로그 저장**: 매 critic 평가 완료 후 `save_loop_log()`를 호출하여 Oracle DB에 중간 행(`is_final=FALSE`)을 INSERT한다. 최종 output/fallback 후 `save_audit_log()`로 최종 행(`is_final=TRUE`)을 추가로 INSERT한다. 따라서 request_id당 **N+1행**이 저장된다 (N=critic 평가 횟수).

### 7.1 조건별 라우팅 (_critic_node)

`ablation_condition` 값에 따라 Self-Corrective Loop 동작이 달라진다.

| 조건 | 이름 | Tier 0 동작 | Tier 소진 처리 |
|------|------|------------|----------------|
| **A** | Proposal System | 자가 교정 + 멀티 티어 에스컬레이션 (기본 동작) | Fallback |
| **E** | Baseline | RAGAS 평가 후 즉시 출력 (루프 없음) | 즉시 output |

### 7.2 tier_path 추적

`state["tier_path"]`는 에스컬레이션 경로를 문자열로 기록한다.

| 값 | 의미 |
|----|------|
| `"0"` | Tier 0에서 완료 |
| `"0→1"` | Tier 0 → Tier 1 에스컬레이션 |
| `"0→1→2"` | Tier 0 → Tier 1 → Tier 2까지 에스컬레이션 |

## 8. output_agent Agent — 최종 출력

**FK Grade 계산 → 한국어 번역 → 출처·면책 조항 추가 → 감사 로그 저장** 순서로 실행된다.

1. **FK Grade 계산**: `output_agent` 호출 *전* 영어 원문 답변(`state["answer"]`)에 대해 `flesch_kincaid_grade_en()`을 실행하여 Flesch-Kincaid Grade Level을 계산한다. 번역 후에는 영어 텍스트가 사라지므로 반드시 번역 전에 계산한다.
2. **한국어 번역**: LLM이 답변 본문을 한국어로 번역한다 (원문이 영·중·일 등일 수 있음).
3. **출처·면책 조항 추가**: 검색 출처(MSD 매뉴얼 파일·페이지 / LLM / 웹 URL)와 면책 조항을 추가한다.
4. **감사 로그 저장**: `save_audit_log(fk_grade=fk)`를 호출하여 Oracle DB에 최종 행(`is_final=TRUE`)을 INSERT한다. `fk_grade`는 번역 전 영어 원문 기준 값이다.

## 9. fallback — 원문 제시

모든 Tier를 소진했을 때 두 단계로 처리한다.

1. **best_answer 우선 사용**: 루프 전체에서 Q_total(`0.4·F + 0.4·AR + 0.2·CP`)이 가장 높은 답변(`best_answer`)이 존재하면 해당 답변을 사용한다. FK Grade를 계산한 후 `output_agent`로 한국어 번역까지 수행한다.
2. **원문 제시 (최후 수단)**: `best_answer`가 없으면 검색된 원문 청크를 그대로 제시하고 `fk_grade=None`으로 저장한다.

완료 후 `save_audit_log(..., is_fallback=True)`를 호출한다. 혼합 언어 답변(원문 제시)은 FK Grade를 계산하지 않으므로 `fk_grade=None`으로 저장한다.
