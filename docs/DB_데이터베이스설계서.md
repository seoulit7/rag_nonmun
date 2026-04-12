# 데이터베이스 설계서 (Database Design Document)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v1.0  
**작성일**: 2026-04-12  
**작성자**: 연구자

---

## 1. 개요

### 1.1 목적

본 문서는 의료 정보 자기교정 RAG 시스템에서 사용하는 데이터베이스의 논리적·물리적 설계를 정의한다. 시스템은 Supabase(PostgreSQL) 클라우드 데이터베이스를 사용하여 모든 질의 처리 이력과 RAGAS 평가 결과를 감사 로그로 저장한다.

### 1.2 데이터베이스 구성

| 항목 | 내용 |
|------|------|
| **DBMS** | PostgreSQL (Supabase 클라우드) |
| **연결 방식** | psycopg2-binary 직접 연결 (DSN URL) |
| **스키마** | public |
| **테이블 수** | 1개 (`rag_audit_log`) |
| **로컬 스토리지** | FAISS 인덱스 파일 (`db/msd_faiss.index`) |

> **참고**: FAISS 벡터 인덱스는 관계형 DB가 아닌 로컬 바이너리 파일로 관리하며, 본 문서의 논리/물리 모델은 PostgreSQL 감사 로그 테이블을 대상으로 한다.

---

## 2. 논리적 데이터 모델 (Logical Data Model)

### 2.1 엔터티 정의

시스템에서 식별된 핵심 엔터티는 다음과 같다.

#### 엔터티: 질의 요청 (Request)

사용자가 시스템에 제출한 하나의 의료 정보 질의 단위.

| 속성명 | 설명 | 타입 | 필수 |
|--------|------|------|------|
| 요청ID | 워크플로우 전체를 식별하는 고유 UUID | UUID | ✅ |
| 사용자수준 | 자동 분류된 사용자 유형 | 문자열 | ✅ |
| 원본질문 | 사용자가 입력한 한국어 원문 질문 | 텍스트 | ✅ |
| 최종답변 | 한국어로 번역된 최종 답변 | 텍스트 | - |
| LLM모델 | 사용된 LLM 백엔드 | 문자열 | - |
| 생성일시 | 요청이 처리된 시각 | 타임스탬프 | - |

#### 엔터티: 평가 루프 (Evaluation Loop)

하나의 질의 요청 내에서 Tier별·루프별로 발생하는 각 RAGAS 평가 이벤트.

| 속성명 | 설명 | 타입 | 필수 |
|--------|------|------|------|
| 로그ID | 평가 이벤트 고유 식별자 | 정수 | ✅ |
| 최적화쿼리 | 해당 루프에서 사용된 영문 검색 쿼리 | 텍스트 | - |
| 검색계층 | 검색 소스 계층 (0=VectorDB, 1=LLM, 2=Web) | 정수 | ✅ |
| 루프횟수 | 해당 Tier 내 재시도 횟수 (0부터 시작) | 정수 | ✅ |
| Faithfulness | 답변의 컨텍스트 근거성 점수 | 실수(0~1) | - |
| AnswerRelevance | 답변과 질문의 관련성 점수 | 실수(0~1) | - |
| ContextPrecision | 검색된 청크의 유효성 점수 | 실수(0~1) | - |
| 에스컬레이션여부 | 이 평가 후 상위 Tier로 에스컬레이션됐는지 | 불리언 | - |
| Fallback여부 | 모든 Tier 소진으로 Fallback 처리됐는지 | 불리언 | - |
| 검색문서수 | 검색된 컨텍스트 청크 수 | 정수 | - |
| 평가소요시간 | RAGAS 평가 수행 시간 (밀리초) | 정수 | - |

### 2.2 엔터티-관계 다이어그램 (ERD)

```
┌──────────────────────────────────────────┐
│            질의 요청 (Request)            │
├──────────────────────────────────────────┤
│ PK  요청ID         UUID    NOT NULL      │
│     사용자수준      VARCHAR NOT NULL      │
│     원본질문        TEXT    NOT NULL      │
│     최종답변        TEXT    NULL          │
│     LLM모델         VARCHAR NULL          │
│     생성일시        TIMESTAMPTZ NULL      │
└──────────────────────┬───────────────────┘
                       │ 1
                       │ (하나의 요청은 1개 이상의
                       │  평가 루프를 가진다)
                       │ N
┌──────────────────────▼───────────────────┐
│           평가 루프 (Evaluation Loop)     │
├──────────────────────────────────────────┤
│ PK  로그ID          BIGINT  NOT NULL      │
│ FK  요청ID          UUID    NOT NULL      │
│     최적화쿼리      TEXT    NULL          │
│     검색계층        INTEGER NOT NULL      │
│     루프횟수        INTEGER NOT NULL      │
│     Faithfulness    FLOAT   NULL (0~1)   │
│     AnswerRelevance FLOAT   NULL (0~1)   │
│     ContextPrecision FLOAT  NULL (0~1)   │
│     에스컬레이션여부 BOOLEAN NULL         │
│     Fallback여부    BOOLEAN NULL          │
│     검색문서수      INTEGER NULL          │
│     평가소요시간    INTEGER NULL          │
└──────────────────────────────────────────┘
```

### 2.3 비즈니스 규칙

| 규칙 ID | 규칙 내용 |
|---------|-----------|
| BR-01 | 사용자수준은 'Professional' 또는 'Consumer' 중 하나여야 한다 |
| BR-02 | 검색계층(tier_id)은 0, 1, 2 중 하나여야 한다 |
| BR-03 | 루프횟수(loop_count)는 0 이상 3 이하여야 한다 |
| BR-04 | Faithfulness, AnswerRelevance, ContextPrecision은 0.0~1.0 범위여야 한다 |
| BR-05 | 하나의 요청ID에 대해 여러 행이 존재할 수 있으며, 각 행은 특정 Tier·루프의 평가 결과를 나타낸다 |
| BR-06 | final_answer는 최초 INSERT 시 NULL이며, output/fallback 완료 후 UPDATE된다 |
| BR-07 | is_escalated=True인 행은 해당 루프에서 상위 Tier로 전환이 결정됐음을 의미한다 |
| BR-08 | is_fallback=True인 행은 모든 Tier 소진 후 Fallback으로 처리된 최종 행이다 |

### 2.4 데이터 흐름 시나리오

#### 시나리오 A: Tier 0 단일 성공 (1개 행 생성)

```
request_id = "abc-001"

| log_id | tier_id | loop_count | ragas_f | ragas_ar | ragas_cp | is_escalated | is_fallback |
|--------|---------|-----------|---------|----------|----------|--------------|-------------|
|   58   |    0    |     0     |  0.930  |  0.921   |  0.881   |    False     |    False    |
```

#### 시나리오 B: Tier 0 재시도 후 Tier 2까지 에스컬레이션 (3개 행 생성)

```
request_id = "xyz-002"

| log_id | tier_id | loop_count | ragas_f | ragas_ar | ragas_cp | is_escalated | is_fallback |
|--------|---------|-----------|---------|----------|----------|--------------|-------------|
|   68   |    0    |     0     |  0.349  |  0.190   |  0.150   |    True      |    False    |  ← AR<0.3 즉시 에스컬레이션
|   69   |    1    |     1     |  0.584  |  0.540   |  0.480   |    True      |    False    |  ← Tier1 기준 미달, Tier2로
|   70   |    2    |     2     |  0.821  |  0.894   |  0.887   |    False     |    False    |  ← 최종 성공
```

---

## 3. 물리적 데이터 모델 (Physical Data Model)

### 3.1 테이블 정의: `rag_audit_log`

```sql
CREATE TABLE public.rag_audit_log (
    log_id              BIGINT          NOT NULL,
    request_id          UUID            NOT NULL DEFAULT gen_random_uuid(),
    user_level          VARCHAR(20)     NOT NULL,
    original_query      TEXT            NOT NULL,
    optimized_query     TEXT,
    final_answer        TEXT,
    tier_id             INTEGER         NOT NULL,
    loop_count          INTEGER         NOT NULL,
    ragas_f             DOUBLE PRECISION,
    ragas_ar            DOUBLE PRECISION,
    ragas_cp            DOUBLE PRECISION,
    is_escalated        BOOLEAN                  DEFAULT false,
    is_fallback         BOOLEAN                  DEFAULT false,
    retrieved_doc_count INTEGER,
    llm_model           VARCHAR(50),
    execution_time_ms   INTEGER,
    created_at          TIMESTAMPTZ              DEFAULT now()
);
```

### 3.2 컬럼 상세 명세

| 컬럼명 | 데이터 타입 | NULL 허용 | 기본값 | 설명 |
|--------|------------|----------|--------|------|
| `log_id` | BIGINT | NOT NULL | — | 행 고유 식별자 (PK, 자동 증가) |
| `request_id` | UUID | NOT NULL | `gen_random_uuid()` | 워크플로우 전체 고유 ID. 동일 요청의 여러 루프는 같은 request_id 공유 |
| `user_level` | VARCHAR(20) | NOT NULL | — | 사용자 수준. 'Professional' 또는 'Consumer' |
| `original_query` | TEXT | NOT NULL | — | 사용자가 입력한 한국어 원본 질문 |
| `optimized_query` | TEXT | NULL | — | 해당 루프에서 LLM이 생성한 영문 최적화 검색 쿼리 |
| `final_answer` | TEXT | NULL | — | 최종 한국어 번역 답변. INSERT 시 NULL, output/fallback 후 UPDATE |
| `tier_id` | INTEGER | NOT NULL | — | 검색 계층. 0=VectorDB(FAISS), 1=LLM 학습데이터, 2=웹검색(DuckDuckGo) |
| `loop_count` | INTEGER | NOT NULL | — | 해당 Tier 내 재시도 순번. 0부터 시작, 최대 3 |
| `ragas_f` | DOUBLE PRECISION | NULL | — | RAGAS Faithfulness 점수 (0.0~1.0). 답변의 컨텍스트 근거성 |
| `ragas_ar` | DOUBLE PRECISION | NULL | — | RAGAS Answer Relevance 점수 (0.0~1.0). 답변-질문 관련성 |
| `ragas_cp` | DOUBLE PRECISION | NULL | — | RAGAS Context Precision 점수 (0.0~1.0). 검색 청크 유효성 |
| `is_escalated` | BOOLEAN | NULL | `false` | 이 행에서 상위 Tier로 에스컬레이션 결정 여부 |
| `is_fallback` | BOOLEAN | NULL | `false` | 모든 Tier 소진으로 Fallback 처리된 최종 행 여부 |
| `retrieved_doc_count` | INTEGER | NULL | — | 검색된 컨텍스트 청크 수 |
| `llm_model` | VARCHAR(50) | NULL | — | 사용된 LLM 백엔드 식별자 ('openai', 'gemini') |
| `execution_time_ms` | INTEGER | NULL | — | RAGAS 평가 소요 시간 (밀리초) |
| `created_at` | TIMESTAMPTZ | NULL | `now()` | 행 생성 시각 (UTC 기준 자동 기록) |

### 3.3 기본 키 (Primary Key)

```sql
ALTER TABLE public.rag_audit_log
    ADD CONSTRAINT rag_audit_log_pkey PRIMARY KEY (log_id);
```

| 제약명 | 대상 컬럼 | 유형 |
|--------|----------|------|
| `rag_audit_log_pkey` | `log_id` | PRIMARY KEY (UNIQUE INDEX) |

### 3.4 체크 제약조건 (Check Constraints)

```sql
-- 사용자 수준 도메인 제약
ALTER TABLE public.rag_audit_log
    ADD CONSTRAINT rag_audit_log_user_level_check
    CHECK (user_level IN ('Professional', 'Consumer'));

-- 검색 계층 도메인 제약
ALTER TABLE public.rag_audit_log
    ADD CONSTRAINT rag_audit_log_tier_id_check
    CHECK (tier_id = ANY (ARRAY[0, 1, 2]));

-- 루프 횟수 범위 제약
ALTER TABLE public.rag_audit_log
    ADD CONSTRAINT rag_audit_log_loop_count_check
    CHECK (loop_count >= 0 AND loop_count <= 3);

-- RAGAS 점수 범위 제약 (0.0 ~ 1.0)
ALTER TABLE public.rag_audit_log
    ADD CONSTRAINT rag_audit_log_ragas_f_check
    CHECK (ragas_f >= 0.0 AND ragas_f <= 1.0);

ALTER TABLE public.rag_audit_log
    ADD CONSTRAINT rag_audit_log_ragas_ar_check
    CHECK (ragas_ar >= 0.0 AND ragas_ar <= 1.0);

ALTER TABLE public.rag_audit_log
    ADD CONSTRAINT rag_audit_log_ragas_cp_check
    CHECK (ragas_cp >= 0.0 AND ragas_cp <= 1.0);
```

| 제약명 | 대상 컬럼 | 조건 |
|--------|----------|------|
| `rag_audit_log_user_level_check` | `user_level` | IN ('Professional', 'Consumer') |
| `rag_audit_log_tier_id_check` | `tier_id` | IN (0, 1, 2) |
| `rag_audit_log_loop_count_check` | `loop_count` | 0 ≤ loop_count ≤ 3 |
| `rag_audit_log_ragas_f_check` | `ragas_f` | 0.0 ≤ ragas_f ≤ 1.0 |
| `rag_audit_log_ragas_ar_check` | `ragas_ar` | 0.0 ≤ ragas_ar ≤ 1.0 |
| `rag_audit_log_ragas_cp_check` | `ragas_cp` | 0.0 ≤ ragas_cp ≤ 1.0 |

### 3.5 인덱스 (Indexes)

```sql
-- request_id 조회용 (단일 요청 전체 루프 조회)
CREATE INDEX idx_request_id
    ON public.rag_audit_log USING btree (request_id);

-- Tier/루프 분석용 (Self-Corrective Loop 패턴 분석)
CREATE INDEX idx_tier_loop
    ON public.rag_audit_log USING btree (tier_id, loop_count);

-- 기간 조회용 (날짜 범위 필터 및 최신순 정렬)
CREATE INDEX idx_created_at
    ON public.rag_audit_log USING btree (created_at);

-- 사용자 수준별 Tier 분석용 (가설 3 성능 시각화)
CREATE INDEX idx_user_tier
    ON public.rag_audit_log USING btree (user_level, tier_id);
```

| 인덱스명 | 대상 컬럼 | 방식 | 용도 |
|---------|----------|------|------|
| `rag_audit_log_pkey` | `log_id` | B-tree (UNIQUE) | PK 조회 |
| `idx_request_id` | `request_id` | B-tree | 단일 요청의 전체 루프 조회 |
| `idx_tier_loop` | `tier_id, loop_count` | B-tree (복합) | Tier별 루프 패턴 분석 |
| `idx_created_at` | `created_at` | B-tree | 기간 필터 조회 및 최신순 정렬 |
| `idx_user_tier` | `user_level, tier_id` | B-tree (복합) | 사용자 수준별 Tier 성능 분석 |

---

## 4. 주요 쿼리 패턴

### 4.1 감사 로그 목록 조회 (페이지네이션)

```sql
SELECT
    log_id,
    request_id,
    created_at AT TIME ZONE 'Asia/Seoul' AS created_at,
    user_level,
    original_query,
    tier_id,
    loop_count,
    ragas_f,
    ragas_ar,
    ragas_cp,
    is_escalated,
    is_fallback,
    execution_time_ms
FROM public.rag_audit_log
WHERE created_at >= '2026-03-01'
  AND created_at <  '2026-04-01'
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;
```

### 4.2 단일 요청 전체 루프 조회

```sql
SELECT *
FROM public.rag_audit_log
WHERE request_id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'::uuid
ORDER BY tier_id ASC, loop_count ASC;
```

### 4.3 성능 시각화용 집계 (요청 기준 최종 결과)

```sql
-- 각 request_id의 최종 행(가장 높은 tier+loop) 기준 성공률 계산
WITH final_rows AS (
    SELECT DISTINCT ON (request_id)
        request_id, tier_id, loop_count,
        ragas_f, ragas_ar, ragas_cp,
        is_escalated, is_fallback, user_level
    FROM public.rag_audit_log
    ORDER BY request_id, tier_id DESC, loop_count DESC
)
SELECT
    COUNT(*)                                             AS total_requests,
    ROUND(AVG(ragas_f)::numeric, 3)                     AS avg_faithfulness,
    ROUND(AVG(ragas_ar)::numeric, 3)                    AS avg_answer_relevance,
    ROUND(AVG(ragas_cp)::numeric, 3)                    AS avg_context_precision,
    SUM(CASE WHEN ragas_f >= 0.8
              AND ragas_ar >= 0.8
              AND ragas_cp >= 0.8 THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*)                              AS success_rate_pct
FROM final_rows;
```

### 4.4 Self-Corrective Loop 효과 분석 (가설 1)

```sql
-- loop_count별 평균 Faithfulness (95% CI 계산용)
SELECT
    loop_count,
    COUNT(*)                            AS n,
    AVG(ragas_f)                        AS mean_f,
    STDDEV(ragas_f)                     AS std_f,
    1.96 * STDDEV(ragas_f) / SQRT(COUNT(*)) AS ci95
FROM public.rag_audit_log
WHERE ragas_f IS NOT NULL
GROUP BY loop_count
ORDER BY loop_count;
```

### 4.5 사용자 수준별 품질 비교 (가설 3)

```sql
-- Professional vs Consumer 최종 행 RAGAS 점수 비교
WITH final_rows AS (
    SELECT DISTINCT ON (request_id)
        user_level, ragas_f, ragas_ar, ragas_cp
    FROM public.rag_audit_log
    ORDER BY request_id, tier_id DESC, loop_count DESC
)
SELECT
    user_level,
    COUNT(*)          AS n,
    AVG(ragas_f)      AS avg_f,
    AVG(ragas_ar)     AS avg_ar,
    AVG(ragas_cp)     AS avg_cp,
    STDDEV(ragas_f)   AS std_f
FROM final_rows
GROUP BY user_level;
```

### 4.6 최종 답변 업데이트

```sql
-- output/fallback 완료 후 final_answer 일괄 업데이트
UPDATE public.rag_audit_log
SET final_answer = '한국어 최종 답변 텍스트'
WHERE request_id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'::uuid;
```

---

## 5. 로컬 파일 스토리지 (FAISS 인덱스)

관계형 DB 외에 벡터 검색을 위한 로컬 파일 스토리지를 사용한다.

| 항목 | 내용 |
|------|------|
| **파일 경로** | `db/msd_faiss.index` |
| **파일 유형** | FAISS IndexFlatL2 바이너리 파일 |
| **생성 시점** | 앱 최초 실행 시 자동 생성 또는 수동 재빌드 |
| **저장 내용** | MSD Manual PDF 청크의 384차원 임베딩 벡터 |
| **임베딩 모델** | `sentence-transformers/all-MiniLM-L6-v2` (384차원) |
| **청크 크기** | 최대 500자, 60자 오버랩 |
| **메타데이터** | 청크 텍스트 및 출처(파일명#페이지)를 별도 pickle로 저장 |

### FAISS 인덱스 구조

```
db/
├── msd_faiss.index       # FAISS 벡터 인덱스 (IndexFlatL2)
└── msd_faiss_meta.pkl    # 청크 텍스트 및 출처 메타데이터
```

---

## 6. 데이터 생명주기

```
[질의 제출]
    │
    ▼
[RAGAS 평가 완료]
    │  INSERT INTO rag_audit_log
    │  (final_answer = NULL)
    ▼
[output / fallback 완료]
    │  UPDATE rag_audit_log
    │  SET final_answer = '한국어 답변'
    │  WHERE request_id = '...'
    ▼
[감사 로그 완성]
    │
    ├─ 대시보드 로그 조회 (fetch_logs / fetch_detail)
    └─ 성능 시각화 집계 (performance_viz._load_data)
```

---

## 7. 데이터 보존 및 관리

| 항목 | 정책 |
|------|------|
| **보존 기간** | 연구 기간 전체 (별도 삭제 정책 없음) |
| **백업** | Supabase 자동 백업 (클라우드 서비스 기본 제공) |
| **접근 권한** | SUPABASE_DB_URL 소유자만 접근 가능 |
| **개인정보** | 사용자 식별 정보 미포함 (request_id는 랜덤 UUID) |

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
