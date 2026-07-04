# 개발 환경 명세서 (Development Environment Specification)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v3.0  
**작성일**: 2026-05-16  
**작성자**: 연구자

---

## 1. 개발 환경 개요

본 시스템은 Python 기반의 LLM 응용 시스템으로, LangGraph 워크플로우, FAISS 벡터 검색, RAGAS 자동 평가, Flesch-Kincaid Grade Level 가독성 측정, Streamlit UI를 핵심 기술 스택으로 한다. 외부 LLM API(OpenAI/Gemini)와 Oracle Database를 활용한다.

---

## 2. 하드웨어 환경

| 항목 | 사양 |
|------|------|
| **운영체제** | Windows 11 Pro (Build 26200) |
| **CPU** | x86-64 아키텍처 |
| **GPU** | 미사용 (CPU 전용 추론) |
| **CUDA** | 비활성화 (torch.cuda.is_available() = False) |

> **비고**: FAISS 인덱싱 및 sentence-transformers 임베딩은 CPU에서 실행된다. GPU 환경에서는 `faiss-gpu` 및 CUDA 지원 PyTorch로 교체하면 성능 향상 가능.

---

## 3. 소프트웨어 환경

### 3.1 언어 및 런타임

| 항목 | 버전 |
|------|------|
| **Python** | 3.11.9 |
| **패키지 관리** | pip / Poetry |
| **가상환경** | 권장 (venv 또는 conda) |

> Python 3.11은 asyncio 성능 개선 및 TypedDict 지원이 안정적이며, LangGraph·RAGAS의 권장 버전이다.

---

## 4. 핵심 라이브러리 버전

### 4.1 UI 및 설정

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `streamlit` | 1.55.0 | 웹 UI 프레임워크 |
| `python-dotenv` | 1.2.2 | `.env` 환경변수 로드 |
| `pydantic` | 2.13.0b2 | 데이터 모델 검증 |

### 4.2 LangChain / LangGraph

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `langgraph` | 0.6.11 | StateGraph 기반 LLM 워크플로우 |
| `langchain` | 0.3.28 | LangChain 코어 |
| `langchain-core` | 0.3.83 | 프롬프트, 파서, 메시지 추상화 |
| `langchain-community` | 0.3.31 | 커뮤니티 도구 및 통합 |
| `langchain-openai` | 0.3.35 | OpenAI / Gemini ChatOpenAI 래퍼 |
| `langchain-text-splitters` | 0.3.11 | RecursiveCharacterTextSplitter |
| `langchain-huggingface` | 0.3.1 | HuggingFace 임베딩 연동 |

### 4.3 LLM API

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `openai` | 1.109.1 | OpenAI API 클라이언트 (ChatGPT, RAGAS용 AsyncOpenAI 포함) |
| `tiktoken` | 0.12.0 | OpenAI 토크나이저 |
| `tenacity` | 8.5.0 | API 재시도 로직 |

### 4.4 RAG / 벡터 검색

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `ragas` | 0.4.3 | RAGAS 공식 평가 프레임워크 (Faithfulness / AR / CP) |
| `faiss-cpu` | 1.13.2 | Facebook AI 유사도 검색 (CPU 버전) |
| `sentence-transformers` | 3.4.1 | **BAAI/bge-base-en-v1.5** 임베딩 모델 (768차원) |
| `huggingface-hub` | 0.36.2 | HuggingFace 모델 다운로드 |
| `torch` | 2.2.0 | PyTorch (sentence-transformers 의존성) |
| `numpy` | 1.26.4 | 수치 연산 (벡터 처리) |
| `scikit-learn` | — | 유사도 계산 보조 |

### 4.5 PDF 처리 및 OCR

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `pymupdf` (fitz) | 1.27.2.2 | PDF 텍스트 추출 및 페이지 렌더링 |
| `pypdf` | 4.3.1 | PDF 메타데이터 처리 보조 |
| `rapidocr-onnxruntime` | 1.4.4 | 스캔 PDF OCR (ONNX 런타임 기반) |

### 4.6 웹검색

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `duckduckgo-search` | 8.1.1 | Tier 2 웹검색 (DuckDuckGo API) |

### 4.7 데이터베이스

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `oracledb` | — | Oracle Database 직접 연결 (감사 로그 저장) |

### 4.8 데이터 처리 및 시각화

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `pandas` | 2.3.3 | 감사 로그 데이터 처리 및 집계 |
| `matplotlib` | 3.10.8 | 정적 차트 렌더링 (영어 텍스트 전용 — 폰트 렌더링 안정성) |
| `seaborn` | 0.13.2 | 통계 시각화 (박스플롯, 막대차트 등) |
| `scipy` | 1.17.1 | KDE 분포 계산 (`gaussian_kde`) |

---

## 5. 외부 서비스

### 5.1 LLM API

| 서비스 | 모델 | 역할 | 필수 여부 |
|--------|------|------|----------|
| **OpenAI API** | gpt-4o | RAG 엔진 (답변 생성) | 필수 (기본) |
| **OpenAI API** | gpt-4o-mini | 사용자 분류, 쿼리 최적화, 번역, RAGAS 평가 | 필수 (기본) |
| **Google Gemini API** | gemini-2.5-pro | RAG 엔진 (OpenAI 대체) | 선택 |
| **Google Gemini API** | gemini-2.5-flash | 분류, 최적화, 번역, 평가 (OpenAI 대체) | 선택 |

> Gemini는 OpenAI 호환 API(`https://generativelanguage.googleapis.com/v1beta/openai/`)를 통해 `ChatOpenAI` 래퍼로 동일하게 사용한다.

### 5.2 데이터베이스

| 서비스 | 유형 | 용도 |
|--------|------|------|
| **Oracle Database** | Oracle DB | 감사 로그(`rag_audit_log`) 저장 및 대시보드 조회. N+1행 설계, fk_grade 컬럼 포함 |

### 5.3 임베딩 모델

| 모델 | 차원 | 다운로드 경로 | 용도 |
|------|------|--------------|------|
| `BAAI/bge-base-en-v1.5` | 768 | HuggingFace Hub | PDF 청크 및 쿼리 벡터화 (코사인 유사도) |

> `config/settings.py`의 기본값은 `sentence-transformers/all-MiniLM-L6-v2`이지만, `.env`의 `MEDICAL_RAG_EMBEDDING_MODEL`로 재정의한다. 현재 배포는 BAAI/bge-base-en-v1.5를 사용하며, 이 모델로 빌드된 FAISS 인덱스가 `db/msd_faiss.index/`에 커밋되어 있다.

---

## 6. 환경 설정 파일

### 6.1 `.env` 파일 구조

```env
# ── LLM API 키 ──────────────────────────────
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...                         # Gemini 사용 시 필수

# ── 모델 설정 ────────────────────────────────
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.5-pro
MEDICAL_RAG_TRANSLATE_MODEL=gpt-4o-mini
MEDICAL_RAG_RAGAS_LLM_MODEL=gpt-4o-mini
MEDICAL_RAG_CLASSIFIER_MODEL=gpt-4o-mini

# ── 데이터베이스 ─────────────────────────────
ORACLE_USER=<사용자명>
ORACLE_PASSWORD=<비밀번호>
ORACLE_DSN=<host>:<port>/<service_name>

# ── 경로 설정 ────────────────────────────────
MEDICAL_RAG_DATA_DIR=data
MEDICAL_RAG_INDEX_PATH=db/msd_faiss.index

# ── 임베딩 모델 ──────────────────────────────
MEDICAL_RAG_EMBEDDING_MODEL=sentence-transformers/BAAI/bge-base-en-v1.5

# ── Self-Corrective Loop 임계값 ──────────────
MEDICAL_RAG_FAITHFULNESS_THRESHOLD=0.8
MEDICAL_RAG_AR_THRESHOLD=0.8
MEDICAL_RAG_CP_THRESHOLD=0.8
MEDICAL_RAG_MAX_LOOPS=3

# ── 즉시 에스컬레이션 임계값 ─────────────────
MEDICAL_RAG_CRITICAL_AR_THRESHOLD=0.3
MEDICAL_RAG_CRITICAL_F_THRESHOLD=0.3
MEDICAL_RAG_CRITICAL_CP_THRESHOLD=0.2

# ── 청크 분할 파라미터 ───────────────────────
MEDICAL_RAG_CHUNK_MAX_CHARS=500
MEDICAL_RAG_CHUNK_OVERLAP=60
MEDICAL_RAG_TOP_K=2

# ── RAGAS 파라미터 ───────────────────────────
MEDICAL_RAG_RAGAS_STRICTNESS=3
MEDICAL_RAG_RAGAS_ANSWER_MAX_CHARS=1500
MEDICAL_RAG_RAGAS_CONTEXT_MAX_CHARS=2000
MEDICAL_RAG_RAGAS_LLM_MAX_TOKENS=8192

# ── OCR 설정 ─────────────────────────────────
MEDICAL_RAG_PDF_OCR=true                     # 스캔 PDF OCR 활성화

# ── 웹검색 ───────────────────────────────────
MEDICAL_RAG_WEB_SEARCH_MAX_RESULTS=3

# ── 로그 수준 ────────────────────────────────
MEDICAL_RAG_LOG_LEVEL=INFO
```

### 6.2 주요 디렉토리 사전 준비

```
rag_nonmun/
├── data/          # MSD Manual PDF 파일 배치
└── db/            # FAISS 인덱스 (msd_faiss.index/ 폴더로 저장)
    └── msd_faiss.index/
        ├── index.faiss   # FAISS 벡터 인덱스 바이너리
        └── index.pkl     # 청크 텍스트 및 출처 메타데이터
```

---

## 7. 설치 및 실행 방법

### 7.1 패키지 설치

```bash
# pip 사용
pip install -r requirements.txt

# Poetry 사용
poetry install
```

### 7.2 환경변수 설정

```bash
# .env 파일 생성 후 API 키 입력
cp .env.example .env
# OPENAI_API_KEY, ORACLE_USER/PASSWORD/DSN 필수 입력
```

### 7.3 Oracle 테이블 생성

```sql
-- db/create_table.sql 스크립트를 Oracle SQL Developer 또는 sqlplus에서 실행
-- docs/DB_데이터베이스설계서.md 섹션 3.1 참조
```

### 7.4 데이터 준비

```
1. MSD Manual PDF 파일을 data/ 폴더에 배치
2. 사이드바 "인덱스 전체 재빌드" 버튼으로 FAISS 인덱스 빌드
   또는 BAAI/bge-base-en-v1.5로 빌드된 기존 인덱스를 db/ 폴더에 배치
   (앱 시작 시 기존 인덱스가 있으면 로드만 수행, 자동 재빌드 없음)
```

### 7.5 앱 실행

```bash
# Streamlit 직접 실행
streamlit run app.py

# launch.py 스크립트 사용
python launch.py
```

### 7.6 성능 평가 일괄 실험

```bash
# Jupyter Notebook으로 실행 (2조건 × 240질문 = 480건 자동 실험)
jupyter notebook main.ipynb
```

---

## 8. 개발 도구

| 도구 | 버전/종류 | 용도 |
|------|-----------|------|
| **IDE** | VSCode | 코드 편집, 디버깅 |
| **버전관리** | Git | 소스 코드 관리 |
| **패키지관리** | pip / Poetry | 의존성 관리 |
| **DB 클라이언트** | Oracle SQL Developer / DBeaver | 감사 로그 조회 및 관리 |
| **실험 환경** | Jupyter Notebook | 시스템 성능 평가 실험 (main.ipynb) |

---

## 9. 기술 스택 요약

```
┌─────────────────────────────────────────────────────────┐
│                   기술 스택 요약                           │
├──────────────┬──────────────────────────────────────────┤
│ 언어         │ Python 3.11.9                            │
│ UI           │ Streamlit 1.55.0                         │
│ 워크플로우   │ LangGraph 0.6.11                         │
│ LLM          │ OpenAI gpt-4o / Gemini 2.5-pro           │
│ LLM 프레임워크│ LangChain 0.3.x                         │
│ 임베딩       │ BAAI/bge-base-en-v1.5 (768차원)          │
│ 벡터DB       │ FAISS 1.13.2 (CPU, 코사인 유사도)        │
│ 평가 (품질)  │ RAGAS 0.4.3 (F / AR / CP / Q_total)      │
│ 평가 (가독성)│ Flesch-Kincaid Grade Level               │
│              │   (Consumer ≤9 / Professional ≥12)       │
│ PDF 처리     │ PyMuPDF 1.27.2 + RapidOCR 1.4.4         │
│ 웹검색       │ DuckDuckGo Search 8.1.1                  │
│ 데이터베이스 │ Oracle Database (oracledb)               │
│              │   N+1행 설계 / fk_grade 컬럼 포함         │
│ 시각화       │ Matplotlib 3.10 / Seaborn 0.13           │
│              │   (matplotlib: 영어 텍스트 전용)           │
│ 딥러닝       │ PyTorch 2.2.0 (CPU)                     │
│ 실험 환경    │ Jupyter Notebook (성능 평가 실험)          │
└──────────────┴──────────────────────────────────────────┘
```

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
