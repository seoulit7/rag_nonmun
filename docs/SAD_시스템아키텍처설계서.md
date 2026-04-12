# 시스템 아키텍처 설계서 (System Architecture Design)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v1.0  
**작성일**: 2026-04-12  
**작성자**: 연구자

---

## 1. 문서 개요

본 문서는 LangGraph 기반 의료 정보 자기교정 RAG 시스템의 전체 아키텍처를 정의한다. 시스템의 컴포넌트 구성, 계층 구조, 데이터 흐름, 모듈 간 의존관계, 핵심 설계 결정사항을 포함한다.

---

## 2. 아키텍처 개요

### 2.1 아키텍처 스타일

본 시스템은 다음 아키텍처 패턴을 복합적으로 적용한다:

| 패턴 | 적용 범위 |
|------|-----------|
| **그래프 기반 워크플로우** | LangGraph StateGraph를 이용한 에이전트 파이프라인 |
| **계층형 아키텍처** | UI → 비즈니스 로직(Graph) → 인프라 → 외부 API |
| **Self-Corrective Loop** | RAGAS 평가 결과에 따라 동적으로 라우팅되는 피드백 루프 |
| **다중 계층 폴백** | Tier 0 → Tier 1 → Tier 2 → Fallback 순의 계단식 에스컬레이션 |

### 2.2 전체 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────────┐
│                          사용자 (Browser)                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  HTTP (Streamlit)
┌───────────────────────────────▼─────────────────────────────────────┐
│                       Presentation Layer                             │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────┐  │
│  │  app.py      │  │  ui/sidebar   │  │  ui/dashboard            │  │
│  │  (진입점)    │  │  (설정 패널)  │  │  ├ log_viewer            │  │
│  │              │  │               │  │  └ performance_viz       │  │
│  └──────┬───────┘  └───────────────┘  └──────────────────────────┘  │
└─────────┼───────────────────────────────────────────────────────────┘
          │  run_medical_self_corrective_rag()
┌─────────▼───────────────────────────────────────────────────────────┐
│                      Business Logic Layer                            │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              LangGraph StateGraph (graph.py)                  │  │
│  │                                                               │  │
│  │  [level_classifier] ──► [query_rewriter] ──► [rag_engine]    │  │
│  │         │                      ▲                  │           │  │
│  │         │                      │ (Self-Corrective │           │  │
│  │         │                      │      Loop)       ▼           │  │
│  │         │                  [critic] ◄──────────────           │  │
│  │         │                      │                              │  │
│  │         │               ┌──────┴──────┐                       │  │
│  │         │               ▼             ▼                       │  │
│  │         │           [output]      [fallback]                  │  │
│  │         │               │             │                       │  │
│  │         └───────────────┴─────────────┘                       │  │
│  │                         END                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ agents/      │  │ core/        │  │ config/settings.py      │   │
│  │ classifier   │  │ llm_client   │  │ (모든 임계값·모델 설정)   │   │
│  │ rewriter     │  │ (OpenAI/     │  └─────────────────────────┘   │
│  │ rag_engine   │  │  Gemini)     │                                 │
│  │ critic       │  └──────────────┘                                 │
│  │ output       │                                                    │
│  └──────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────┐
│                       Infrastructure Layer                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ infra/           │  │ infra/           │  │ tools/           │  │
│  │ vector_store.py  │  │ audit_logger.py  │  │ vector_search.py │  │
│  │ (FAISS 인덱스)   │  │ (감사 로그 저장) │  │ web_search.py   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
└───────────┼─────────────────────┼─────────────────────┼────────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼────────────┐
│                        External Services                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
│  │ FAISS Index  │  │  Supabase    │  │  OpenAI API  │  │DuckDuck│  │
│  │ (로컬 파일)  │  │  PostgreSQL  │  │  Gemini API  │  │Go API  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 계층별 상세 설계

### 3.1 Presentation Layer (UI)

| 모듈 | 파일 | 역할 |
|------|------|------|
| 진입점 | `app.py` | Streamlit 앱 초기화, 세션 상태 관리, 라우팅 |
| 사이드바 | `ui/sidebar.py` | 페르소나 선택, LLM 백엔드 선택, 인덱스 재빌더, 대시보드 메뉴 |
| 헤더 | `ui/header.py` | 시스템 설명 및 플로우 안내 |
| PDF 업로더 | `ui/pdf_uploader.py` | PDF 업로드 및 증분 인덱싱 UI |
| 스텝 렌더러 | `ui/step_renderers.py` | LangGraph 노드별 실시간 상태 표시 |
| 점수 카드 | `ui/score_card.py` | F/AR/CP 점수 카드 렌더링 |
| 결과 패널 | `ui/result_panel.py` | 최종 답변 표시 |
| 로그 뷰어 | `ui/dashboard/log_viewer.py` | 감사 로그 목록/상세 조회 |
| 성능 시각화 | `ui/dashboard/performance_viz.py` | 가설별 시각화 차트 |

**Streamlit 세션 상태 관리:**

```python
SESSION_DEFAULTS = {
    "result": "",
    "logs": [],
    "scores": {},
    "detected_level": "",
    "search_tier": 0,
    "llm_provider": "openai",
    "dashboard_menu": "",
}
```

---

### 3.2 Business Logic Layer (LangGraph 워크플로우)

#### 3.2.1 GraphState 설계

시스템 전체를 관통하는 단일 상태 객체. 모든 노드는 이 상태를 읽고 업데이트한다.

```python
class GraphState(TypedDict):
    request_id:               str          # 요청 고유 UUID
    question:                 str          # 원본 한국어 질문
    user_level:               str          # "Professional" | "Consumer"
    queries:                  List[str]    # 최적화된 영문 쿼리 이력
    context:                  List[str]    # 검색된 컨텍스트 청크
    context_sources:          List[str]    # 출처 메타데이터
    answer:                   str          # 현재 생성된 영문 답변
    critic_score:             float        # Faithfulness (0~1)
    answer_relevance_score:   float        # Answer Relevance (0~1)
    context_precision_score:  float        # Context Precision (0~1)
    hallucination_flags:      List[str]    # 할루시네이션 감지 플래그
    search_tier:              int          # 0=VectorDB, 1=LLM, 2=Web
    llm_provider:             str          # "openai" | "gemini"
    loop_count:               int          # 현재 Tier 재시도 횟수
    log:                      List[str]    # 실행 로그
```

#### 3.2.2 LangGraph 노드 설계

```
level_classifier ──► query_rewriter ──► rag_engine ──► critic
                           ▲                              │
                           │        (Self-Corrective)     │ F≥0.8∧AR≥0.8∧CP≥0.8
                           └──────────────────────────────┤
                                                          │ 기준 미달
                                                     ┌────▼────┐
                                                     │ output  │──► END
                                                     └─────────┘
                                                     ┌──────────┐
                                                     │ fallback │──► END
                                                     └──────────┘
```

| 노드 | 함수 | 역할 |
|------|------|------|
| `level_classifier` | `agents/classifier.py` | LLM으로 사용자 수준 분류 (Professional/Consumer) |
| `query_rewriter` | `agents/rewriter.py` | 한국어 질문 → 영문 의료 검색 쿼리 최적화 |
| `rag_engine` | `agents/rag_engine.py` | Tier별 검색 및 답변 합성 (ReAct 에이전트) |
| `critic` | `agents/critic.py` + `graph.py` | RAGAS 3중 평가 + Self-Corrective 라우팅 |
| `output` | `agents/output.py` | 영문 답변 → 한국어 번역 + 출처·면책 조항 추가 |
| `fallback` | `graph.py` | 모든 Tier 소진 시 원문 제시 |

#### 3.2.3 Self-Corrective Loop 라우팅 로직

```
critic 노드 평가 후:

[성공 조건]
F ≥ 0.8 AND AR ≥ 0.8 AND CP ≥ 0.8
  └─► output → END

[Tier 0 즉시 에스컬레이션]
AR < 0.3  OR  (F < 0.3 AND CP < 0.2)
  └─► search_tier = 1, loop_count = 0 → query_rewriter

[Tier 0 재시도]
기준 미달 AND loop_count < MAX_LOOPS(3)
  └─► loop_count + 1 → query_rewriter (동일 Tier 재검색)

[Tier 0 → Tier 1 에스컬레이션]
loop_count >= MAX_LOOPS(3)
  └─► search_tier = 1, loop_count = 0 → query_rewriter

[Tier 1 → Tier 2 에스컬레이션]
Tier 1 기준 미달
  └─► search_tier = 2, loop_count = 0 → query_rewriter

[Fallback]
Tier 2 기준 미달 (모든 Tier 소진)
  └─► fallback → END
```

#### 3.2.4 RAG Engine Tier별 동작

| Tier | 검색 소스 | 에이전트 방식 | 시스템 프롬프트 | Temperature |
|------|-----------|--------------|----------------|-------------|
| **0** | FAISS VectorDB (MSD Manual) | ReAct 에이전트 + `search_msd_manual` 도구 | 엄격 컨텍스트 한정 (컨텍스트 외 정보 금지) | 0.0 |
| **1** | LLM 학습데이터 | 도구 없음, LLM 직접 생성 | 의료 지식 기반 답변 | 0.1 |
| **2** | DuckDuckGo 웹검색 | ReAct 에이전트 + `search_web` 도구 | 웹 정보 합성 허용 | 0.1 |

---

### 3.3 Infrastructure Layer

#### 3.3.1 벡터 스토어 (`infra/vector_store.py`)

```
PDF 파일 (data/)
    │
    ▼
[PyMuPDF (fitz)]  텍스트 추출
    │
    ├─ 텍스트 있음 → 직접 사용
    └─ 텍스트 없음 (스캔 PDF) → [RapidOCR] → 텍스트 인식
    │
    ▼
[RecursiveCharacterTextSplitter]
  chunk_size=500, chunk_overlap=60
    │
    ▼
[URL 청크 필터링]  (http://, https:// 포함 청크 제거)
    │
    ▼
[sentence-transformers/all-MiniLM-L6-v2]  임베딩
    │
    ▼
[FAISS IndexFlatL2]  벡터 인덱스 저장
  db/msd_faiss.index
```

**증분 인덱싱**: 시작 시 이미 인덱싱된 파일 목록과 비교하여 신규 파일만 추가 처리

#### 3.3.2 감사 로거 (`infra/audit_logger.py`)

- **저장 시점**: `_critic_node()` 내 RAGAS 평가 직후
- **업데이트 시점**: `output_agent` 또는 `fallback_node` 완료 후 (`final_answer` UPDATE)
- **스레드 안전**: `threading.local()`로 스레드별 psycopg2 커넥션 관리 (Streamlit 멀티스레드 대응)

```sql
-- rag_audit_log 테이블 핵심 컬럼
request_id        UUID        -- 워크플로우 전체 고유 ID
user_level        VARCHAR     -- Professional | Consumer
original_query    TEXT        -- 원본 한국어 질문
optimized_query   TEXT        -- 최적화된 영문 쿼리
final_answer      TEXT        -- 최종 한국어 답변 (UPDATE로 채움)
tier_id           INTEGER     -- 0 | 1 | 2
loop_count        INTEGER     -- 현재 Tier 재시도 횟수
ragas_f           FLOAT       -- Faithfulness
ragas_ar          FLOAT       -- Answer Relevance
ragas_cp          FLOAT       -- Context Precision
is_escalated      BOOLEAN     -- 이번 행에서 에스컬레이션 발생 여부
is_fallback       BOOLEAN     -- 최종 Fallback 여부
execution_time_ms INTEGER     -- RAGAS 평가 소요 시간(ms)
created_at        TIMESTAMPTZ -- 자동 기록
```

#### 3.3.3 RAGAS 평가 (`infra/evaluator.py`)

Streamlit 이벤트 루프 충돌 문제를 해결하기 위한 설계:

```
Streamlit 메인 스레드
    │  (자체 asyncio 이벤트 루프 보유)
    │
    ▼
ThreadPoolExecutor (max_workers=1)
    │  별도 워커 스레드
    ▼
asyncio.new_event_loop()  (새 이벤트 루프 생성)
    │
    ▼
asyncio.gather(
    faith.ascore(),   # Faithfulness
    arel.ascore(),    # Answer Relevance
    cpre.ascore()     # Context Precision
)
    │  timeout=120초
    ▼
OfficialRagasScores(faithfulness, answer_relevance, context_precision, hallu_flags)
```

---

### 3.4 Core Layer

#### 3.4.1 LLM 클라이언트 (`core/llm_client.py`)

**듀얼 LLM 백엔드 설계**: ContextVar를 이용한 스레드-안전 provider 전환

```
ContextVar[llm_provider]  ← "openai" | "gemini"
    │
    ├─ "openai"  → ChatOpenAI(model=gpt-4o, api_key=OPENAI_API_KEY)
    └─ "gemini"  → ChatOpenAI(
                       model=gemini-2.5-pro,
                       base_url=GEMINI_OPENAI_COMPAT_BASE_URL,  # OpenAI 호환 API
                       api_key=GEMINI_API_KEY
                   )
```

**모델 역할 분리:**

| 역할 | OpenAI 모델 | Gemini 모델 |
|------|-------------|-------------|
| 사용자 분류 | gpt-4o-mini | gemini-2.5-flash |
| 쿼리 최적화 | gpt-4o-mini | gemini-2.5-flash |
| RAG 엔진 (답변 생성) | gpt-4o | gemini-2.5-pro |
| 번역 | gpt-4o-mini | gemini-2.5-flash |
| RAGAS 평가 | gpt-4o-mini | gemini-2.5-flash |

---

## 4. 데이터 흐름 설계

### 4.1 정상 흐름 (Tier 0 성공)

```
사용자 질문 (한국어)
    │
    ▼
[level_classifier]
  LLM → user_level: "Consumer" | "Professional"
    │
    ▼
[query_rewriter]
  LLM → queries: ["common cold symptoms consumer"]
    │
    ▼
[rag_engine - Tier 0]
  ReAct 에이전트 → search_msd_manual("common cold symptoms consumer")
  → context: [청크1, 청크2, 청크3]
  → answer: "[Consumer Summary] Common cold symptoms include..."
    │
    ▼
[critic]
  RAGAS 평가 (ThreadPoolExecutor)
  → critic_score(F): 0.91
  → answer_relevance_score(AR): 0.88
  → context_precision_score(CP): 0.85
  → F≥0.8 AND AR≥0.8 AND CP≥0.8 → output으로 라우팅
  → audit_log INSERT (is_escalated=False, is_fallback=False)
    │
    ▼
[output]
  LLM 번역 → answer: "감기의 주요 증상은..."
  → audit_log UPDATE (final_answer)
    │
    ▼
최종 답변 (한국어) + 점수 카드 표시
```

### 4.2 에스컬레이션 흐름 (Tier 0 → Tier 1 → Tier 2)

```
[critic] AR=0.19 < 0.3 (즉시 에스컬레이션)
  → audit_log INSERT (is_escalated=True)
  → search_tier: 0 → 1
    │
    ▼
[query_rewriter] 재최적화
    │
    ▼
[rag_engine - Tier 1]
  LLM 학습데이터 직접 생성
    │
    ▼
[critic] F=0.61 < 0.8 (Tier 1 기준 미달)
  → audit_log INSERT (is_escalated=True)
  → search_tier: 1 → 2
    │
    ▼
[rag_engine - Tier 2]
  DuckDuckGo 웹검색 → answer 합성
    │
    ▼
[critic] F=0.84, AR=0.82, CP=0.81 → 성공
  → audit_log INSERT (is_escalated=False, is_fallback=False)
    │
    ▼
[output] 번역 → END
```

---

## 5. 컴포넌트 의존관계

```
app.py
  └─ ui/sidebar.py
  └─ ui/step_renderers.py
  └─ ui/score_card.py
  └─ ui/result_panel.py
  └─ ui/pdf_uploader.py
  └─ graph.py (run_medical_self_corrective_rag)
       └─ agents/classifier.py
       │    └─ core/llm_client.py
       │    └─ utils/json_parser.py
       └─ agents/rewriter.py
       │    └─ core/llm_client.py
       └─ agents/rag_engine.py
       │    └─ core/llm_client.py
       │    └─ tools/vector_search.py
       │    │    └─ infra/vector_store.py
       │    └─ tools/web_search.py
       └─ agents/critic.py
       │    └─ infra/evaluator.py
       │         └─ core/llm_client.py (RAGAS용)
       └─ agents/output.py
       │    └─ core/llm_client.py
       └─ infra/audit_logger.py (psycopg2 → Supabase)
       └─ config/settings.py (모든 임계값·모델명)

ui/dashboard/performance_viz.py
  └─ psycopg2 → Supabase (직접 쿼리)
  └─ matplotlib, seaborn, scipy
```

---

## 6. 설정 관리 설계

모든 임계값과 모델명은 `config/settings.py`에서 환경변수로 중앙 관리한다.

| 설정 항목 | 환경변수 | 기본값 | 용도 |
|-----------|---------|--------|------|
| `FAITHFULNESS_THRESHOLD` | `MEDICAL_RAG_FAITHFULNESS_THRESHOLD` | `0.8` | Self-Corrective Loop 성공 기준 |
| `AR_THRESHOLD` | `MEDICAL_RAG_AR_THRESHOLD` | `0.8` | Self-Corrective Loop 성공 기준 |
| `CP_THRESHOLD` | `MEDICAL_RAG_CP_THRESHOLD` | `0.8` | Self-Corrective Loop 성공 기준 |
| `CRITICAL_AR_THRESHOLD` | `MEDICAL_RAG_CRITICAL_AR_THRESHOLD` | `0.3` | 즉시 에스컬레이션 조건 (AR) |
| `CRITICAL_F_THRESHOLD` | `MEDICAL_RAG_CRITICAL_F_THRESHOLD` | `0.3` | 즉시 에스컬레이션 조건 (F) |
| `CRITICAL_CP_THRESHOLD` | `MEDICAL_RAG_CRITICAL_CP_THRESHOLD` | `0.2` | 즉시 에스컬레이션 조건 (CP) |
| `MAX_LOOPS` | `MEDICAL_RAG_MAX_LOOPS` | `3` | Tier당 최대 재시도 횟수 |
| `OPENAI_MODEL` | `OPENAI_MODEL` | `gpt-4o` | RAG 엔진 모델 |
| `GEMINI_MODEL` | `GEMINI_MODEL` | `gemini-2.5-pro` | Gemini RAG 엔진 모델 |
| `CHUNK_MAX_CHARS` | `MEDICAL_RAG_CHUNK_MAX_CHARS` | `500` | 청크 최대 길이 |
| `CHUNK_OVERLAP` | `MEDICAL_RAG_CHUNK_OVERLAP` | `60` | 청크 오버랩 크기 |
| `RAG_TOP_K` | `MEDICAL_RAG_TOP_K` | `2` | VectorDB 검색 상위 K개 |
| `PDF_OCR_ENABLED` | `MEDICAL_RAG_PDF_OCR` | `false` | 스캔 PDF OCR 활성화 |

---

## 7. 디렉토리 구조

```
rag_nonmun/
├── app.py                    # Streamlit 진입점
├── graph.py                  # LangGraph 그래프 빌드 및 실행
├── medical_rag_graph.py      # 하위 호환 re-export 모듈
├── launch.py                 # CLI 실행 스크립트
│
├── agents/                   # LangGraph 노드 에이전트
│   ├── classifier.py         # 사용자 수준 분류
│   ├── rewriter.py           # 쿼리 최적화
│   ├── rag_engine.py         # 검색 및 답변 합성 (Tier 0/1/2)
│   ├── critic.py             # RAGAS 평가 + 라우팅 판단
│   └── output.py             # 번역 및 최종 답변 생성
│
├── core/
│   └── llm_client.py         # LLM 클라이언트 (OpenAI/Gemini 듀얼 지원)
│
├── infra/
│   ├── vector_store.py       # FAISS 인덱스 빌드 및 관리
│   ├── evaluator.py          # RAGAS 공식 메트릭 평가
│   └── audit_logger.py       # Supabase 감사 로그 저장
│
├── tools/
│   ├── vector_search.py      # FAISS 검색 도구 (LangChain Tool)
│   └── web_search.py         # DuckDuckGo 웹검색 도구
│
├── models/
│   └── state.py              # GraphState TypedDict 정의
│
├── config/
│   └── settings.py           # 전체 환경변수 설정 중앙 관리
│
├── ui/
│   ├── constants.py          # SESSION_DEFAULTS, TIER_CONFIGS
│   ├── sidebar.py            # 사이드바 UI
│   ├── header.py             # 헤더 및 안내
│   ├── pdf_uploader.py       # PDF 업로드 UI
│   ├── step_renderers.py     # LangGraph 단계별 실시간 렌더링
│   ├── score_card.py         # RAGAS 점수 카드
│   ├── result_panel.py       # 최종 답변 패널
│   └── dashboard/
│       ├── log_viewer.py     # 로그 조회 화면
│       ├── log_list.py       # 로그 목록 컴포넌트
│       ├── log_detail.py     # 로그 상세 컴포넌트
│       ├── log_query.py      # 로그 DB 쿼리
│       └── performance_viz.py # 성능 시각화 대시보드
│
├── utils/
│   └── json_parser.py        # LLM JSON 응답 파싱 유틸리티
│
├── data/                     # MSD 매뉴얼 PDF 원본
├── db/                       # FAISS 인덱스 파일
│   └── msd_faiss.index
├── docs/                     # 시스템 산출 문서
├── .env                      # API 키 및 환경변수 (비공개)
├── requirements.txt          # Python 패키지 의존성
└── pyproject.toml            # Poetry 프로젝트 설정
```

---

## 8. 핵심 설계 결정사항

### 8.1 LangGraph StateGraph 채택

**결정**: 순수 Python 파이프라인 대신 LangGraph StateGraph 사용  
**이유**: 조건부 라우팅(Self-Corrective Loop, 에스컬레이션)을 선언적으로 정의할 수 있고, 노드 실행 순서를 그래프 구조로 명확히 표현 가능. `stream_mode="updates"`로 실시간 UI 업데이트 지원.

### 8.2 RAGAS 비동기 평가 격리

**결정**: Streamlit 메인 스레드와 별도로 ThreadPoolExecutor + 새 asyncio 이벤트 루프에서 RAGAS 실행  
**이유**: Streamlit은 자체 asyncio 이벤트 루프를 보유하며, `asyncio.run()`을 직접 호출 시 "This event loop is already running" 오류 발생. 별도 스레드에서 새 루프 생성으로 충돌 방지.

### 8.3 영문 쿼리 최적화

**결정**: 사용자 한국어 질문을 영문으로 변환하여 FAISS 검색  
**이유**: MSD Manual PDF가 전부 영어로 작성되어 있으므로, 한국어 쿼리로 검색 시 임베딩 유사도가 낮음. 영문 의료 학술 용어로 변환하여 검색 정확도 향상.

### 8.4 Tier별 시스템 프롬프트 분리

**결정**: Tier 0은 컨텍스트 외 정보 생성을 엄격히 금지하고, Tier 2는 자유 합성 허용  
**이유**: Tier 0에서 할루시네이션 방지가 최우선. LLM이 컨텍스트에 없는 의료 정보를 생성하면 RAGAS Faithfulness 점수가 낮아져 자동으로 재시도 또는 에스컬레이션됨.

### 8.5 psycopg2 직접 연결

**결정**: ORM 대신 psycopg2로 Supabase PostgreSQL 직접 연결  
**이유**: 감사 로그는 단순 INSERT/UPDATE 패턴이며, ORM 도입 시 의존성이 늘어남. 스레드별 커넥션 관리로 Streamlit 멀티스레드 환경에 대응.

### 8.6 OCR 선택적 활성화

**결정**: `PDF_OCR_ENABLED` 환경변수로 OCR 기능 On/Off 제어  
**이유**: RapidOCR은 ONNX 런타임을 사용하므로 실행 시간이 길어짐. 텍스트 PDF만 처리하는 환경에서는 OCR을 비활성화하여 인덱싱 속도 향상.

---

## 9. 보안 및 운영 설계

### 9.1 API 키 관리

- 모든 API 키는 `.env` 파일에만 저장, 소스코드 하드코딩 금지
- `python-dotenv`로 환경변수 로드, 환경변수 미설정 시 빈 문자열 반환

### 9.2 오류 처리 전략

| 오류 유형 | 처리 방식 |
|-----------|-----------|
| LLM API 오류 | `max_retries=6` 자동 재시도 |
| RAGAS 평가 실패 | 각 메트릭별 개별 try/except, 실패 시 0.0 반환 |
| RAGAS 타임아웃 | `future.result(timeout=120)` 초과 시 0.0 반환 |
| DB 커넥션 오류 | 커넥션 초기화 후 로그만 기록, 시스템 계속 실행 |
| 모든 Tier 소진 | Fallback 노드로 라우팅, 원문 제시 |

### 9.3 싱글턴 그래프 인스턴스

LangGraph 그래프는 모듈 수준 싱글턴으로 관리하여 매 요청마다 재컴파일하지 않는다:

```python
_compiled_graph = None

def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
```

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
