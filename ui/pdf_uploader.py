from pathlib import Path

import streamlit as st

import config.settings as settings
from tools.vector_search import add_pdfs_to_vector_db


def _list_existing_pdfs() -> list:
    """data 폴더에 있는 PDF 파일 목록을 반환한다."""
    data_path = Path(settings.DATA_DIR)
    if not data_path.exists():
        return []
    return sorted(data_path.glob("**/*.pdf"))


def render_pdf_uploader() -> None:
    """사이드바에 PDF 업로드 및 인덱스 재빌드 UI를 렌더링한다."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 PDF 문서 관리")

    # 현재 등록된 PDF 목록
    existing = _list_existing_pdfs()
    if existing:
        with st.sidebar.expander(f"등록된 PDF ({len(existing)}개)", expanded=False):
            for p in existing:
                st.caption(f"• {p.name}")
    else:
        st.sidebar.caption("등록된 PDF가 없습니다.")

    # 파일 업로더
    uploaded = st.sidebar.file_uploader(
        "PDF 업로드 (복수 선택 가능)",
        type=["pdf"],
        accept_multiple_files=True,
        help="업로드 후 '인덱스 재빌드' 버튼을 눌러 FAISS DB를 갱신하세요.",
    )

    if not uploaded:
        return

    st.sidebar.caption(f"{len(uploaded)}개 파일 선택됨")

    if not st.sidebar.button("🔄 인덱스 재빌드", type="primary", use_container_width=True):
        return

    # 진행 상황 표시 (메인 화면 중앙에 표시)
    st.markdown("---")
    st.subheader("📊 FAISS 인덱스 재빌드")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(percent: int, message: str) -> None:
        progress_bar.progress(percent)
        status_text.info(f"**{percent}%** — {message}")

    try:
        added_files, added_chunks = add_pdfs_to_vector_db(uploaded, on_progress=on_progress)
        progress_bar.progress(100)
        if added_files == 0:
            status_text.warning("⚠️ 모두 이미 등록된 파일입니다. 새로 추가된 PDF가 없습니다.")
            st.sidebar.warning("추가된 PDF 없음")
        else:
            status_text.success(
                f"✅ {added_files}개 PDF, {added_chunks}개 청크가 기존 DB에 추가되었습니다."
            )
            st.sidebar.success(f"✅ {added_files}개 PDF 추가 완료")
        # 다음 질문 시 새 DB가 자동 사용됨 (initialize_vector_db는 _db=None 감지)
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"❌ 재빌드 실패: {e}")
        st.sidebar.error("인덱스 재빌드 실패")
