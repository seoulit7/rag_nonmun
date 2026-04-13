from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config.settings as settings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def load_pdf_docs(pdf_path: str) -> List[Document]:
    """단일 PDF 파일을 로드하여 Document 목록을 반환한다.

    MEDICAL_RAG_PDF_OCR=true 인 경우:
    - 텍스트 레이어가 없는 페이지는 PyMuPDF로 이미지 렌더링 후
      rapidocr-onnxruntime으로 OCR을 수행한다.
    - 텍스트 레이어가 있는 페이지는 그대로 사용한다.
    """
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(pdf_path)

    if not settings.PDF_OCR_ENABLED:
        return PyPDFLoader(str(path)).load()

    # OCR 모드: 텍스트가 없는 페이지만 rapidocr로 추출
    # PyMuPDF + rapidocr 미설치 환경(Streamlit Cloud 등)은 일반 PyPDFLoader로 fallback
    try:
        import fitz  # PyMuPDF
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    except Exception:
        # OCR 라이브러리 없는 환경 → 일반 텍스트 추출로 fallback
        return PyPDFLoader(str(path)).load()

    try:
        doc = fitz.open(str(path))
        documents: List[Document] = []

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()

            if not text:
                # 스캔 페이지: 이미지로 렌더링 후 OCR
                try:
                    mat = fitz.Matrix(2, 2)  # 2× 확대로 인식률 향상
                    pix = page.get_pixmap(matrix=mat)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 4:
                        arr = arr[:, :, :3]

                    result, _ = _ocr_engine(arr)
                    if result:
                        text = " ".join(line[1] for line in result)
                except Exception:
                    pass

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_num},
                    )
                )

        return documents

    except Exception:
        # fitz 처리 실패 시 일반 PyPDFLoader로 fallback
        return PyPDFLoader(str(path)).load()


def load_and_split_pdfs(folder_path: str) -> List[Document]:
    """폴더 내 전체 PDF를 로드하고 청크로 분할한다."""
    pdf_paths = sorted(Path(folder_path).glob("**/*.pdf"))
    if not pdf_paths:
        raise RuntimeError(
            f"'{folder_path}' 폴더에 PDF 파일이 없습니다. "
            "MSD 매뉴얼 PDF를 data 폴더에 넣은 후 다시 실행하세요."
        )

    docs: List[Document] = []
    for path in pdf_paths:
        docs.extend(load_pdf_docs(str(path)))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_MAX_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_faiss_db(docs: List[Document]) -> FAISS:
    """Document 목록으로 FAISS DB를 생성합니다."""
    return FAISS.from_documents(docs, _get_embeddings())


def save_faiss_db(db: FAISS, index_path: str) -> None:
    """FAISS DB를 로컬에 저장합니다."""
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    db.save_local(index_path)


def load_faiss_db(index_path: str) -> FAISS:
    """로컬에 저장된 FAISS DB를 로드합니다."""
    return FAISS.load_local(
        index_path,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve(db: FAISS, query: str, top_k: int = None) -> List[Document]:
    """FAISS DB에서 유사도 높은 상위 top_k 문서를 반환합니다."""
    k = top_k or settings.RAG_TOP_K
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
