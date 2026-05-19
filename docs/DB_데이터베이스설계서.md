# 데이터베이스 설계서 (Database Design Document)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v4.0  
**작성일**: 2026-05-16  
**작성자**: 연구자

---

## 1. 개요

### 1.1 목적

본 문서는 의료 정보 자기교정 RAG 시스템에서 사용하는 데이터베이스의 논리적·물리적 설계를 정의한다. 시스템은 Supabase(PostgreSQL) 클라우드 데이터베이스를 사용하여 모든 질의 처리 이력과 RAGAS 평가 결과를 감사 로그로 저장한다.

**버전별 변경 이력:**

| 버전 | 주요 변경사항 |
|------|--------------|
| v1.0 | 루프마다 INSERT + 최종 UPDATE 설계 |
| v2.0 | 완료 후 단 1회 INSERT (tier_path, self_correction_count 추가) |
| v3.0 | 평가마다 INSERT + 완료 시 1회 INSERT (loop_number, is_final 추가) |
| v4.0 | `fk_grade` 컬럼 추가 (Flesch-Kincaid Grade Level, is_final=TRUE 행만 기록) |

### 1.2 데이터베이스 구성

| 항목 | 내용 |
|------|------|
| **DBMS** | PostgreSQL (Supabase 클라우드) |
| **연결 방식** | psycopg2-binary 직접 연결 (DSN URL) |
| **스키마** | public |
| **테이블 수** | 1개 (`rag_audit_log`) |
| **로컬 스토리지** | FAISS 인덱스 폴더 (`db/msd_faiss.index/`) |

> **참고**: FAISS 벡터 인덱스는 관계형 DB가 아닌 로컬 바이너리 파일로 관리하며, 본 문서의 논리/물리 모델은 PostgreSQL 감사 로그 테이블을 대상으로 한다.

---

## 2. 논리적 데이터 모델 (Logical Data Model)

### 2.1 핵심 설계 원칙: 평가 회차마다 1행 (v3.0), fk_grade 추가 (v4.0)

매 critic 평가 후 **즉시 1행 INSERT(is_final=FALSE)**하고, output/fallback 완료 시 **추가로 1행 INSERT(is_final=TRUE)**한다.

| v1.0 (구) | v2.0 | v3.0 | v4.0 (현재) |
|-----------|------|------|-------------|
| 루프마다 INSERT + 최종 UPDATE | 완료 후 단 1회 INSERT | 평가마다 INSERT + 완료 시 1회 INSERT | v3.0 + `fk_grade` 컬럼 추가 |
| tier_id + loop_count 컬럼 | final_tier + tier_path 컬럼 | loop_number + is_final 컬럼 추가 | is_final=TRUE 행에만 fk_grade 기록 |
| request_id당 N행 | request_id당 1행 | request_id당 N+1행 (N=평가 횟수) | 동일 |
| final_answer = NULL → UPDATE | final_answer 포함 단일 INSERT | is_final=FALSE: 중간 점수 / is_final=TRUE: 최종 답변 포함 | is_final=TRUE: fk_grade 포함 |

**저장 흐름**:
- `_critic_node` 완료마다 → `save_loop_log()` → is_final=**FALSE**, loop_number=eval_count, final_answer=NULL, fk_grade=NULL
- `_output_node` 완료 시 → `save_audit_log(fk_grade=fk)` → is_final=**TRUE**, final_answer 포함, **fk_grade 포함**
- `_fallback_node` 완료 시 → `save_audit_log(fk_grade=None)` → is_final=**TRUE**, is_fallback=TRUE, fk_grade=NULL

**fk_grade 계산 시점**: `graph.py`의 `_output_node`에서 `output_agent` 호출(한국어 번역) *전*에 영어 원문으로 `flesch_kincaid_grade_en()`을 실행한다. 번역 후에는 영어 텍스트가 소실되므로 반드시 번역 전에 계산해야 한다.

### 2.2 엔터티 정의

#### 엔터티: 감사 로그 (Audit Log)

하나의 의료 질의 요청에 대한 처리 결과를 담는 레코드 (request_id당 N+1행).

| 속성명 | 설명 | 타입 | 필수 |
|--------|------|------|------|
| 로그ID | 행 고유 식별자 | BIGSERIAL | ✅ |
| 요청ID | 워크플로우 전체 UUID | UUID | ✅ |
| 생성일시 | INSERT 시각 (UTC) | TIMESTAMPTZ | ✅ |
| **루프번호** | critic 평가 누적 횟수 (1부터 시작) | INTEGER | ✅ |
| **최종여부** | 최종 출력 완료 행이면 TRUE | BOOLEAN | ✅ |
| **Ablation 조건** | 실험 조건 ('A'~'E'), 일반 운영 시 NULL | CHAR(1) | - |
| **질문 번호** | STQS-40 질문 순번 (1-40), 일반 운영 시 NULL | SMALLINT | - |
| **질환명** | STQS-40 대상 질환, 일반 운영 시 NULL | VARCHAR | - |
| **정답 레이블** | STQS-40 수준 정답 ('P'/'C'), 일반 운영 시 NULL | CHAR(1) | - |
| 사용자수준 | 분류된 사용자 유형 | VARCHAR | ✅ |
| 원본질문 | 한국어 원문 질문 | TEXT | ✅ |
| 최적화쿼리 | 최종 영문 검색 쿼리 | TEXT | - |
| **예상 티어** | STQS-40 기대 도달 티어 (0/1/2), 일반 운영 시 NULL | SMALLINT | - |
| **최종 티어** | 실제 답변이 생성된 검색 계층 | SMALLINT | ✅ |
| **티어 경로** | 에스컬레이션 경로 ("0"/"0→1"/"0→1→2") | VARCHAR | ✅ |
| **에스컬레이션여부** | Tier 이동 발생 여부 | BOOLEAN | ✅ |
| Fallback여부 | Fallback으로 처리된 여부 | BOOLEAN | ✅ |
| **자가교정횟수** | Tier 0 내 Self-Corrective Loop 누적 횟수 | SMALLINT | ✅ |
| Faithfulness | RAGAS F 점수 (0~1) | FLOAT | - |
| AnswerRelevance | RAGAS AR 점수 (0~1) | FLOAT | - |
| ContextPrecision | RAGAS CP 점수 (0~1) | FLOAT | - |
| **Q_total** | 종합 품질 점수 (0.4·F + 0.4·AR + 0.2·CP) | FLOAT | - |
| **할루시네이션감지** | 할루시네이션 패턴 탐지 여부 | BOOLEAN | ✅ |
| **할루시네이션건수** | 탐지된 할루시네이션 항목 수 | SMALLINT | ✅ |
| 검색문서수 | 검색된 컨텍스트 청크 수 | INTEGER | - |
| LLM모델 | 사용된 LLM 백엔드 식별자 | VARCHAR | - |
| 실행시간 | 전체 워크플로우 소요 시간 (ms) | INTEGER | - |
| 최종답변 | 한국어 번역된 최종 답변 | TEXT | - |
| **FK Grade** | Flesch-Kincaid Grade Level (영어 원문 기준) | FLOAT | - |

### 2.3 ERD (단일 테이블)

```
┌────────────────────────────────────────────────────────────┐
│                    감사 로그 (Audit Log)                     │
├────────────────────────────────────────────────────────────┤
│ PK  log_id               BIGSERIAL   NOT NULL              │
│     request_id           UUID        NOT NULL              │
│     created_at           TIMESTAMPTZ NOT NULL              │
│     loop_number          INTEGER     NOT NULL              │
│     is_final             BOOLEAN     NOT NULL DEFAULT false│
│  ── Ablation Study ────────────────────────────────────    │
│     ablation_condition   CHAR(1)     NULL  ('A'~'E')       │
│     query_index          SMALLINT    NULL  (1-40)          │
│     disease              VARCHAR     NULL                  │
│     query_level_label    CHAR(1)     NULL  ('P'/'C')       │
│  ── 요청 ──────────────────────────────────────────────    │
│     user_level           VARCHAR     NOT NULL              │
│     original_query       TEXT        NOT NULL              │
│     optimized_query      TEXT        NULL                  │
│  ── 티어 라우팅 ────────────────────────────────────────    │
│     expected_tier        SMALLINT    NULL  (0/1/2)         │
│     final_tier           SMALLINT    NOT NULL DEFAULT 0    │
│     tier_path            VARCHAR     NOT NULL DEFAULT '0'  │
│     is_escalated         BOOLEAN     NOT NULL DEFAULT false│
│     is_fallback          BOOLEAN     NOT NULL DEFAULT false│
│     self_correction_count SMALLINT   NOT NULL DEFAULT 0    │
│  ── RAGAS 평가 ─────────────────────────────────────────   │
│     ragas_f              FLOAT       NULL  (0~1)           │
│     ragas_ar             FLOAT       NULL  (0~1)           │
│     ragas_cp             FLOAT       NULL  (0~1)           │
│     q_total              FLOAT       NULL  (0~1)           │
│  ── 할루시네이션 ───────────────────────────────────────    │
│     hallucination_detected BOOLEAN   NOT NULL DEFAULT false│
│     hallucination_count  SMALLINT    NOT NULL DEFAULT 0    │
│  ── 시스템 ─────────────────────────────────────────────   │
│     retrieved_doc_count  INTEGER     NULL                  │
│     llm_model            VARCHAR     NULL                  │
│     execution_time_ms    INTEGER     NULL                  │
│     final_answer         TEXT        NULL                  │
│     fk_grade             FLOAT       NULL                  │
└────────────────────────────────────────────────────────────┘
```

### 2.4 비즈니스 규칙

| 규칙 ID | 규칙 내용 |
|---------|-----------|
| BR-01 | user_level은 'Professional' 또는 'Consumer' 중 하나여야 한다 |
| BR-02 | final_tier는 0, 1, 2 중 하나여야 한다 |
| BR-03 | expected_tier는 0, 1, 2 또는 NULL이어야 한다 |
| BR-04 | F, AR, CP, q_total은 0.0~1.0 범위여야 한다 |
| BR-05 | q_total = 0.4·F + 0.4·AR + 0.2·CP (audit_logger가 자동 계산) |
| BR-06 | ablation_condition은 'A'~'E' 또는 NULL(일반 운영)이어야 한다 |
| BR-07 | query_level_label은 'P', 'C' 또는 NULL이어야 한다 |
| BR-08 | request_id당 N+1행이 존재한다 (N=critic 평가 횟수, 중간 N행 + 최종 1행) |
| BR-09 | is_escalated = (tier_path != "0") 로 자동 결정된다 |
| BR-10 | is_final=FALSE 행: critic 노드 완료 직후 저장 (final_answer=NULL, fk_grade=NULL) |
| BR-11 | is_final=TRUE 행: output/fallback 노드 완료 직후 저장 (final_answer 포함) |
| BR-12 | 최종 결과 집계 시 반드시 WHERE is_final = TRUE 필터를 사용한다 |
| BR-13 | fk_grade는 is_final=TRUE이고 is_fallback=FALSE인 행에만 값이 존재한다. 중간 행과 fallback 행은 NULL |
| BR-14 | fk_grade는 한국어 번역 전 영어 원문 답변을 기준으로 계산한다 (Flesch-Kincaid Grade Level 공식) |
| BR-15 | Tier 1 행: F, CP, q_total은 NULL (컨텍스트 없음). AR만 유효 |

### 2.5 데이터 흐름 시나리오

#### 시나리오 A: 조건 D — Tier 0 첫 번째 평가 성공 (재시도 없음)

```
request_id = "abc-001", ablation_condition = "D"
→ 저장 행: 2행 (중간 1행 + 최종 1행)

| loop_number | is_final | final_tier | ragas_f | ragas_ar | q_total | fk_grade | final_answer |
|-------------|----------|------------|---------|----------|---------|----------|--------------|
|      1      |  FALSE   |     0      |  0.930  |  0.921   |  0.908  |   NULL   |    NULL      |  ← save_loop_log
|      1      |  TRUE    |     0      |  0.930  |  0.921   |  0.908  |  11.20   |  "갑상선..."  |  ← save_audit_log
```

#### 시나리오 B: 조건 A — Tier 0 자가 교정 2회 후 Tier 1 에스컬레이션 → 성공

```
request_id = "xyz-002", ablation_condition = "A"
→ 저장 행: 4행 (중간 3행 + 최종 1행)

| loop_number | is_final | final_tier | tier_path | ragas_f | ragas_ar | self_correction_count | fk_grade | final_answer |
|-------------|----------|------------|-----------|---------|----------|-----------------------|----------|--------------|
|      1      |  FALSE   |     0      |    "0"    |  0.61   |  0.57    |           0           |   NULL   |     NULL     |
|      2      |  FALSE   |     0      |    "0"    |  0.72   |  0.68    |           1           |   NULL   |     NULL     |
|      3      |  FALSE   |     1      |   "0→1"   |  NULL   |  0.840   |           2           |   NULL   |     NULL     |  ← Tier1: F/CP=NULL
|      3      |  TRUE    |     1      |   "0→1"   |  NULL   |  0.840   |           2           |   NULL   |  "관상동맥..." |  ← Tier1 최종행: fk_grade=NULL
```

#### 시나리오 C: 조건 A — Tier 0 성공 (Consumer, fk_grade 낮음)

```
request_id = "cons-003", ablation_condition = "A", user_level="Consumer"
→ 저장 행: 2행

| loop_number | is_final | user_level | ragas_f | ragas_ar | q_total | fk_grade | final_answer |
|-------------|----------|------------|---------|----------|---------|----------|--------------|
|      1      |  FALSE   |  Consumer  |  0.88   |  0.85    |  0.872  |   NULL   |     NULL     |
|      1      |  TRUE    |  Consumer  |  0.88   |  0.85    |  0.872  |  8.50    |  "감기란..."  |  ← fk_grade ≤ 9 목표
```

---

## 3. 물리적 데이터 모델 (Physical Data Model)

### 3.1 테이블 정의: `rag_audit_log`

```sql
DROP TABLE IF EXISTS public.rag_audit_log;

CREATE TABLE public.rag_audit_log (
    log_id                  BIGSERIAL       PRIMARY KEY,
    request_id              UUID            NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    -- 루프 추적 (v3.0)
    loop_number             INTEGER         NOT NULL DEFAULT 1,
    is_final                BOOLEAN         NOT NULL DEFAULT false,
    -- Ablation Study 메타데이터
    ablation_condition      CHAR(1)         DEFAULT NULL,
    query_index             SMALLINT        DEFAULT NULL,
    disease                 VARCHAR(50)     DEFAULT NULL,
    query_level_label       CHAR(1)         DEFAULT NULL,
    -- 요청 정보
    user_level              VARCHAR(20)     NOT NULL,
    original_query          TEXT            NOT NULL,
    optimized_query         TEXT,
    -- 티어 라우팅
    expected_tier           SMALLINT        DEFAULT NULL,
    final_tier              SMALLINT        NOT NULL DEFAULT 0,
    tier_path               VARCHAR(20)     DEFAULT '0',
    is_escalated            BOOLEAN         NOT NULL DEFAULT false,
    is_fallback             BOOLEAN         NOT NULL DEFAULT false,
    self_correction_count   SMALLINT        NOT NULL DEFAULT 0,
    -- RAGAS 평가
    ragas_f                 DOUBLE PRECISION,
    ragas_ar                DOUBLE PRECISION,
    ragas_cp                DOUBLE PRECISION,
    q_total                 DOUBLE PRECISION,
    -- 할루시네이션
    hallucination_detected  BOOLEAN         NOT NULL DEFAULT false,
    hallucination_count     SMALLINT        NOT NULL DEFAULT 0,
    -- 시스템
    retrieved_doc_count     INTEGER,
    llm_model               VARCHAR(50),
    execution_time_ms       INTEGER,
    final_answer            TEXT,
    -- 가독성 지표 (v4.0 추가)
    fk_grade                DOUBLE PRECISION
);
```

### 3.2 컬럼 상세 명세

| 컬럼명 | 데이터 타입 | NULL 허용 | 기본값 | 설명 |
|--------|------------|----------|--------|------|
| `log_id` | BIGSERIAL | NOT NULL | 자동증가 | 행 고유 식별자 (PK) |
| `request_id` | UUID | NOT NULL | — | 워크플로우 전체 고유 ID |
| `created_at` | TIMESTAMPTZ | NOT NULL | `now()` | INSERT 시각 (UTC) |
| `loop_number` | INTEGER | NOT NULL | 1 | critic 평가 누적 회차 (save_audit_log에서는 최종 eval_count) |
| `is_final` | BOOLEAN | NOT NULL | false | TRUE=최종 결과 행(final_answer 포함), FALSE=중간 평가 행 |
| `ablation_condition` | CHAR(1) | NULL | NULL | Ablation Study 조건 ('A'~'E'). NULL=일반 운영 |
| `query_index` | SMALLINT | NULL | NULL | STQS-40 질문 번호 (1-40). NULL=일반 운영 |
| `disease` | VARCHAR(50) | NULL | NULL | STQS-40 질환명. NULL=일반 운영 |
| `query_level_label` | CHAR(1) | NULL | NULL | STQS-40 수준 정답 레이블 ('P'/'C'). NULL=일반 운영 |
| `user_level` | VARCHAR(20) | NOT NULL | — | 'Professional' 또는 'Consumer' |
| `original_query` | TEXT | NOT NULL | — | 사용자가 입력한 한국어 원본 질문 |
| `optimized_query` | TEXT | NULL | — | 마지막 루프에서 사용된 영문 최적화 검색 쿼리 |
| `expected_tier` | SMALLINT | NULL | NULL | STQS-40 예상 티어 (0/1/2). NULL=일반 운영 |
| `final_tier` | SMALLINT | NOT NULL | 0 | 최종 답변이 생성된 검색 계층 (0/1/2) |
| `tier_path` | VARCHAR(20) | NOT NULL | '0' | 에스컬레이션 경로. "0" / "0→1" / "0→1→2" |
| `is_escalated` | BOOLEAN | NOT NULL | false | Tier 에스컬레이션 발생 여부 (tier_path != "0") |
| `is_fallback` | BOOLEAN | NOT NULL | false | 모든 Tier 소진으로 Fallback 처리 여부 |
| `self_correction_count` | SMALLINT | NOT NULL | 0 | Tier 0 내 자가 교정 누적 횟수 |
| `ragas_f` | DOUBLE PRECISION | NULL | — | RAGAS Faithfulness (0~1). Tier 1 행은 NULL |
| `ragas_ar` | DOUBLE PRECISION | NULL | — | RAGAS Answer Relevance (0~1). 모든 Tier 사용 |
| `ragas_cp` | DOUBLE PRECISION | NULL | — | RAGAS Context Precision (0~1). Tier 1 행은 NULL |
| `q_total` | DOUBLE PRECISION | NULL | — | 종합 품질 점수: 0.4·F + 0.4·AR + 0.2·CP. Tier 1 행은 NULL |
| `hallucination_detected` | BOOLEAN | NOT NULL | false | 의료 도메인 할루시네이션 탐지 여부 |
| `hallucination_count` | SMALLINT | NOT NULL | 0 | 탐지된 할루시네이션 항목 수 |
| `retrieved_doc_count` | INTEGER | NULL | — | 검색된 컨텍스트 청크 수 |
| `llm_model` | VARCHAR(50) | NULL | — | 사용된 LLM 백엔드 ('openai', 'gemini') |
| `execution_time_ms` | INTEGER | NULL | — | 전체 워크플로우 소요 시간 (ms). is_final=FALSE 행은 NULL |
| `final_answer` | TEXT | NULL | — | 한국어 번역된 최종 답변. is_final=FALSE 행은 NULL |
| `fk_grade` | DOUBLE PRECISION | NULL | — | Flesch-Kincaid Grade Level. is_final=TRUE이고 is_fallback=FALSE인 행만 기록. Consumer 목표 ≤9, Professional 목표 ≥12 |

### 3.3 인덱스 (Indexes)

```sql
CREATE INDEX idx_audit_ablation    ON public.rag_audit_log (ablation_condition);
CREATE INDEX idx_audit_request     ON public.rag_audit_log (request_id);
CREATE INDEX idx_audit_disease     ON public.rag_audit_log (disease);
CREATE INDEX idx_audit_query_index ON public.rag_audit_log (query_index);
CREATE INDEX idx_audit_final       ON public.rag_audit_log (request_id, is_final);
```

| 인덱스명 | 대상 컬럼 | 용도 |
|---------|----------|------|
| `rag_audit_log_pkey` | `log_id` | PK 조회 |
| `idx_audit_ablation` | `ablation_condition` | 조건별 성능 비교 집계 |
| `idx_audit_request` | `request_id` | 단일 요청 조회 |
| `idx_audit_disease` | `disease` | 질환별 분석 |
| `idx_audit_query_index` | `query_index` | STQS-40 질문 번호별 분석 |
| `idx_audit_final` | `(request_id, is_final)` | 최종 결과만 조회 시 성능 향상 |

---

## 4. 주요 쿼리 패턴

### 4.1 Ablation Study 조건별 성능 비교

```sql
SELECT
    ablation_condition,
    COUNT(*)                                        AS n,
    ROUND(AVG(ragas_f)::numeric,  3)                AS avg_f,
    ROUND(AVG(ragas_ar)::numeric, 3)                AS avg_ar,
    ROUND(AVG(ragas_cp)::numeric, 3)                AS avg_cp,
    ROUND(AVG(q_total)::numeric,  3)                AS avg_q,
    ROUND(AVG(self_correction_count)::numeric, 2)   AS avg_loops,
    SUM(CASE WHEN is_fallback THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*)                          AS fallback_rate_pct
FROM public.rag_audit_log
WHERE ablation_condition IS NOT NULL
  AND is_final = TRUE
GROUP BY ablation_condition
ORDER BY ablation_condition;
```

### 4.2 사용자 수준 분류 정확도 (Ablation 조건별)

```sql
SELECT
    ablation_condition,
    query_level_label                               AS answer_label,
    user_level                                      AS predicted_level,
    COUNT(*)                                        AS n,
    SUM(CASE
        WHEN (query_level_label = 'P' AND user_level = 'Professional')
          OR (query_level_label = 'C' AND user_level = 'Consumer')
        THEN 1 ELSE 0 END) * 100.0 / COUNT(*)       AS accuracy_pct
FROM public.rag_audit_log
WHERE query_level_label IS NOT NULL
  AND is_final = TRUE
GROUP BY ablation_condition, query_level_label, user_level
ORDER BY ablation_condition, query_level_label;
```

### 4.3 티어 라우팅 정확도 (예상 vs 실제)

```sql
SELECT
    expected_tier,
    final_tier,
    COUNT(*)                                        AS n,
    SUM(CASE WHEN expected_tier = final_tier THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*)                          AS tier_accuracy_pct
FROM public.rag_audit_log
WHERE expected_tier IS NOT NULL
  AND is_final = TRUE
GROUP BY expected_tier, final_tier
ORDER BY expected_tier, final_tier;
```

### 4.4 할루시네이션 감소율 (조건 A vs E 비교)

```sql
SELECT
    ablation_condition,
    COUNT(*)                                                    AS n,
    SUM(hallucination_count)                                    AS total_hallucinations,
    ROUND(AVG(hallucination_count)::numeric, 3)                 AS avg_hallu_per_request,
    SUM(CASE WHEN hallucination_detected THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*)                                      AS hallu_rate_pct
FROM public.rag_audit_log
WHERE ablation_condition IN ('A', 'E')
  AND is_final = TRUE
GROUP BY ablation_condition;
```

### 4.5 Self-Corrective Loop 효과 (자가 교정 횟수별 품질)

```sql
SELECT
    self_correction_count,
    COUNT(*)                            AS n,
    ROUND(AVG(ragas_f)::numeric,  3)    AS avg_f,
    ROUND(AVG(ragas_ar)::numeric, 3)    AS avg_ar,
    ROUND(AVG(q_total)::numeric,  3)    AS avg_q
FROM public.rag_audit_log
WHERE ablation_condition = 'A'
  AND is_final = TRUE
GROUP BY self_correction_count
ORDER BY self_correction_count;
```

### 4.6 FK Grade 분포 (수준별 가독성 목표 달성률)

```sql
SELECT
    user_level,
    COUNT(*)                                            AS n,
    ROUND(AVG(fk_grade)::numeric, 3)                    AS avg_fk_grade,
    ROUND(STDDEV(fk_grade)::numeric, 3)                 AS std_fk_grade,
    -- Consumer 목표: fk_grade ≤ 9
    SUM(CASE WHEN user_level = 'Consumer'     AND fk_grade <= 9  THEN 1 ELSE 0 END)
        * 100.0 / NULLIF(COUNT(*), 0)                   AS within_target_pct,
    -- Professional 목표: fk_grade ≥ 12
    SUM(CASE WHEN user_level = 'Professional' AND fk_grade >= 12 THEN 1 ELSE 0 END)
        * 100.0 / NULLIF(COUNT(*), 0)                   AS within_target_pct_pro
FROM public.rag_audit_log
WHERE is_final = TRUE
  AND is_fallback = FALSE
  AND fk_grade IS NOT NULL
GROUP BY user_level;
```

### 4.7 감사 로그 목록 조회 (페이지네이션)

```sql
SELECT
    log_id,
    request_id,
    created_at AT TIME ZONE 'Asia/Seoul'    AS created_at_kst,
    ablation_condition,
    user_level,
    original_query,
    final_tier,
    tier_path,
    ragas_f, ragas_ar, ragas_cp, q_total,
    fk_grade,
    is_escalated, is_fallback, execution_time_ms
FROM public.rag_audit_log
WHERE is_final = TRUE
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;
```

### 4.8 단일 요청 상세 조회

```sql
SELECT *
FROM public.rag_audit_log
WHERE request_id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'::uuid
ORDER BY loop_number;
```

---

## 5. 로컬 파일 스토리지 (FAISS 인덱스)

관계형 DB 외에 벡터 검색을 위한 로컬 파일 스토리지를 사용한다.

| 항목 | 내용 |
|------|------|
| **저장 경로** | `db/msd_faiss.index/` (폴더) |
| **파일 구성** | `index.faiss` (벡터 인덱스 바이너리) + `index.pkl` (청크 메타데이터) |
| **생성 방법** | 사이드바 "인덱스 전체 재빌드" 버튼 또는 pre-built 인덱스 배치 |
| **로드 정책** | 앱 시작 시 기존 인덱스가 있으면 로드만 수행 (자동 재빌드 없음) |
| **임베딩 모델** | `BAAI/bge-base-en-v1.5` (768차원, 코사인 유사도) |
| **청크 크기** | 최대 500자, 60자 오버랩 |
| **URL 필터링** | http://, https:// 포함 청크는 인덱싱 제외 |

### FAISS 인덱스 구조

```
db/
└── msd_faiss.index/
    ├── index.faiss    # LangChain FAISS 벡터 인덱스 (IndexFlatL2 + bge-base 정규화 벡터 = 코사인 동치)
    └── index.pkl      # LangChain Document 객체 (청크 텍스트 + 출처 메타데이터)
```

---

## 6. 데이터 생명주기

```
[질의 제출]
    │
    ▼
[워크플로우 실행]
  level_classifier → query_rewriter → rag_engine → critic
  (Self-Corrective Loop / 에스컬레이션 / Ablation 조건 분기)
    │
    ▼
[critic 평가 완료 (매 회차마다)]
    │  save_loop_log() → INSERT is_final=FALSE
    │  (loop_number, RAGAS 점수, tier 정보 포함 / final_answer=NULL / fk_grade=NULL)
    ▼
[output 완료]
    │  fk_grade = flesch_kincaid_grade_en(영어 원문)  ← 번역 전 계산
    │  output_agent() → 한국어 번역
    │  save_audit_log(fk_grade=fk) → INSERT is_final=TRUE
    │  (final_answer, execution_time_ms, fk_grade 포함)
    │  UPDATE 없음
    ▼
[fallback 완료]
    │  save_audit_log(is_fallback=True, fk_grade=None) → INSERT is_final=TRUE
    │  (fk_grade=NULL: 혼합 언어 답변은 FK 계산 불가)
    ▼
[감사 로그 완성 — request_id당 N+1행]
    │
    ├─ 대시보드 로그 조회 (fetch_logs / fetch_detail)
    ├─ 성능 시각화 집계 (performance_viz._load_data)
    └─ Ablation Study 분석 (조건별 집계 쿼리)
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
