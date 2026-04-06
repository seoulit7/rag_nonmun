from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config.settings as settings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def load_and_split_pdfs(folder_path: str) -> List[Document]:
    """폴더 내 PDF를 로드하고 청크로 분할합니다."""
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load()

    if not docs:
        raise RuntimeError(
            f"'{folder_path}' 폴더에 PDF 파일이 없습니다. "
            "MSD 매뉴얼 PDF를 data 폴더에 넣은 후 다시 실행하세요."
        )

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
