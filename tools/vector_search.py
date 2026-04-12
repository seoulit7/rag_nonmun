import json
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.tools import tool

import config.settings as settings
from infra.vector_store import (
    load_pdf_docs,
    load_and_split_pdfs,
    build_faiss_db,
    save_faiss_db,
    load_faiss_db,
    retrieve,
)

_db = None


def _get_indexed_filenames(db) -> set:
    """FAISS 도큐먼트 스토어에서 이미 인덱싱된 파일명 집합을 반환한다."""
    names = set()
    for doc in db.docstore._dict.values():
        src = doc.metadata.get("source", "")
        if src:
            names.add(Path(src).name)
    return names


def initialize_vector_db(
    data_folder: str = None,
    index_path: str = None,
) -> None:
    """FAISS DB를 로드하거나 PDF로부터 새로 빌드한다.

    - 이미 메모리에 로드된 경우(_db is not None): 스킵
    - 인덱스 파일이 있는 경우: 로드 후 data 폴더의 새 PDF만 증분 추가
    - 인덱스 파일이 없는 경우: data 폴더 전체로 신규 빌드
    """
    global _db
    if _db is not None:
        return

    data_folder = data_folder or settings.DATA_DIR
    index_path = index_path or settings.INDEX_PATH
    data_path = Path(data_folder)

    if Path(index_path).exists():
        _db = load_faiss_db(index_path)

        # data 폴더의 PDF 중 아직 인덱싱되지 않은 파일 확인
        if data_path.exists():
            all_pdfs = {p.name: p for p in data_path.glob("**/*.pdf")}
            indexed = _get_indexed_filenames(_db)
            new_pdf_paths = [p for name, p in all_pdfs.items() if name not in indexed]

            if new_pdf_paths:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.CHUNK_MAX_CHARS,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                )
                new_docs = []
                for path in new_pdf_paths:
                    raw = load_pdf_docs(str(path))
                    chunks = splitter.split_documents(raw)
                    new_docs.extend(
                        c for c in chunks
                        if len(c.page_content.strip()) >= 50
                        and not c.page_content.strip().startswith(("http://", "https://"))
                    )

                if new_docs:
                    new_index = build_faiss_db(new_docs)
                    _db.merge_from(new_index)
                    save_faiss_db(_db, index_path)
        return

    docs = load_and_split_pdfs(data_folder)
    _db = build_faiss_db(docs)
    save_faiss_db(_db, index_path)


def add_pdfs_to_vector_db(
    uploaded_files: List,
    data_folder: str = None,
    index_path: str = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> tuple:
    """업로드된 PDF만 임베딩해 기존 FAISS DB에 증분 추가한다.

    기존 DB가 있으면 새 PDF만 임베딩 후 merge_from()으로 병합하므로
    전체 재빌드보다 훨씬 빠르다. 기존 DB가 없으면 새로 생성한다.

    Args:
        uploaded_files: Streamlit UploadedFile 목록
        data_folder: PDF 저장 경로 (기본값: settings.DATA_DIR)
        index_path: FAISS 인덱스 저장 경로 (기본값: settings.INDEX_PATH)
        on_progress: (percent: int, message: str) 콜백

    Returns:
        (추가된 PDF 수, 추가된 청크 수)
    """
    global _db

    data_folder = data_folder or settings.DATA_DIR
    index_path = index_path or settings.INDEX_PATH
    data_path = Path(data_folder)
    data_path.mkdir(parents=True, exist_ok=True)

    def _notify(percent: int, msg: str) -> None:
        if on_progress:
            on_progress(percent, msg)

    # 1단계: 신규 파일만 필터링 (이미 있는 파일 제외)
    _notify(5, "신규 PDF 확인 중...")
    new_files = []
    skipped = []
    for f in uploaded_files:
        dest = data_path / f.name
        if dest.exists():
            skipped.append(f.name)
        else:
            new_files.append(f)

    if not new_files:
        _notify(100, f"모두 이미 등록된 파일입니다. ({len(skipped)}개 스킵)")
        return 0, 0

    if skipped:
        _notify(10, f"{len(skipped)}개 스킵 (이미 등록됨), {len(new_files)}개 신규 처리")

    # 2단계: 신규 PDF 파일 저장
    _notify(15, "신규 PDF 저장 중...")
    temp_paths = []
    for i, f in enumerate(new_files):
        dest = data_path / f.name
        dest.write_bytes(f.read())
        temp_paths.append(dest)
        pct = 15 + int((i + 1) / len(new_files) * 20)
        _notify(pct, f"저장: {f.name} ({i+1}/{len(new_files)})")

    # 3단계: 신규 PDF만 로드 및 청크 분할
    _notify(37, f"신규 PDF {len(new_files)}개 로드 및 청크 분할 중...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_MAX_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    def _is_useful_chunk(doc) -> bool:
        """URL만 있거나 내용이 너무 짧은 청크를 제외한다."""
        text = doc.page_content.strip()
        if len(text) < 50:
            return False
        if text.startswith("http://") or text.startswith("https://"):
            return False
        return True

    new_docs = []
    for i, path in enumerate(temp_paths):
        raw = load_pdf_docs(str(path))
        chunks = splitter.split_documents(raw)
        useful = [c for c in chunks if _is_useful_chunk(c)]
        new_docs.extend(useful)
        pct = 37 + int((i + 1) / len(temp_paths) * 23)
        _notify(pct, f"분할 완료: {path.name} → {len(useful)}개 청크 (필터 후, {i+1}/{len(temp_paths)})")

    _notify(62, f"총 {len(new_docs)}개 신규 청크 생성. 임베딩 중...")

    # 4단계: 신규 청크만 임베딩
    new_index = build_faiss_db(new_docs)
    _notify(88, "기존 DB에 병합 중...")

    # 5단계: 기존 DB에 merge_from() 으로 병합
    index_path_obj = Path(index_path)
    if _db is not None:
        _db.merge_from(new_index)
    elif index_path_obj.exists():
        _db = load_faiss_db(index_path)
        _db.merge_from(new_index)
    else:
        _db = new_index

    # 6단계: 저장
    _notify(95, "인덱스 저장 중...")
    save_faiss_db(_db, index_path)
    _notify(100, f"완료! {len(new_files)}개 PDF, {len(new_docs)}개 청크 추가되었습니다.")

    return len(new_files), len(new_docs)


def rebuild_full_index(
    data_folder: str = None,
    index_path: str = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> tuple:
    """data 폴더의 전체 PDF를 처음부터 재임베딩하여 FAISS 인덱스를 재빌드한다.

    기존 인덱스 파일을 덮어쓰고 메모리의 _db도 교체한다.

    Args:
        data_folder: PDF 폴더 경로 (기본값: settings.DATA_DIR)
        index_path:  FAISS 인덱스 저장 경로 (기본값: settings.INDEX_PATH)
        on_progress: (percent: int, message: str) 진행상황 콜백

    Returns:
        (처리된 PDF 수, 생성된 청크 수)
    """
    global _db

    data_folder = data_folder or settings.DATA_DIR
    index_path = index_path or settings.INDEX_PATH
    data_path = Path(data_folder)

    def _notify(percent: int, msg: str) -> None:
        if on_progress:
            on_progress(percent, msg)

    # 1단계: PDF 목록 수집
    _notify(2, "PDF 파일 목록 수집 중...")
    pdf_paths = sorted(data_path.glob("**/*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"'{data_folder}' 폴더에 PDF 파일이 없습니다.")

    # 2단계: 전체 PDF 로드 및 청크 분할
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_MAX_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    def _is_useful_chunk(doc) -> bool:
        text = doc.page_content.strip()
        if len(text) < 50:
            return False
        if text.startswith("http://") or text.startswith("https://"):
            return False
        return True

    if settings.PDF_OCR_ENABLED:
        _notify(5, f"총 {len(pdf_paths)}개 PDF 발견. OCR 모드로 로드 중... (시간이 더 걸릴 수 있습니다)")
    else:
        _notify(5, f"총 {len(pdf_paths)}개 PDF 발견. 로드 및 청크 분할 중...")

    all_docs = []
    for i, path in enumerate(pdf_paths):
        raw = load_pdf_docs(str(path))
        chunks = splitter.split_documents(raw)
        useful = [c for c in chunks if _is_useful_chunk(c)]
        all_docs.extend(useful)
        pct = 5 + int((i + 1) / len(pdf_paths) * 50)
        _notify(pct, f"[{i+1}/{len(pdf_paths)}] {path.name} → {len(useful)}청크")

    _notify(57, f"총 {len(all_docs)}개 청크 생성. 임베딩 중... (수 분 소요될 수 있습니다)")

    # 3단계: 전체 임베딩 및 FAISS 빌드
    new_db = build_faiss_db(all_docs)
    _notify(92, "인덱스 저장 중...")

    # 4단계: 저장 및 메모리 교체
    save_faiss_db(new_db, index_path)
    _db = new_db
    _notify(100, f"완료! {len(pdf_paths)}개 PDF, {len(all_docs)}개 청크로 인덱스 재빌드 완료.")

    return len(pdf_paths), len(all_docs)


@tool
def search_msd_manual(query: str) -> str:
    """MSD 매뉴얼 FAISS 벡터 DB에서 관련 의료 정보를 검색합니다.

    전문 의료 참고서(MSD Manual)에서 관련 구절을 의미 검색합니다.
    반드시 영어로 검색 쿼리를 작성하세요.

    Args:
        query: 검색할 영문 의료 쿼리

    Returns:
        JSON 문자열 {"chunks": [...], "sources": [...]}
    """
    global _db
    if _db is None:
        initialize_vector_db()

    docs = retrieve(_db, query)
    if not docs:
        return json.dumps({"chunks": ["No relevant MSD document found for this query."], "sources": []})

    chunks = [doc.page_content for doc in docs]
    sources = [
        f"{doc.metadata.get('source', '')}#p{doc.metadata.get('page', 0)}"
        for doc in docs
    ]
    return json.dumps({"chunks": chunks, "sources": sources})
