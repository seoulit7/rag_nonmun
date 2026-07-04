from pathlib import Path

import streamlit as st

import config.settings as settings
from tools.vector_search import add_pdfs_to_vector_db


def _list_existing_pdfs() -> list:
    data_path = Path(settings.DATA_DIR)
    if not data_path.exists():
        return []
    return sorted(data_path.glob("**/*.pdf"))


def render_pdf_uploader() -> None:
    """Render the PDF upload and index rebuild UI in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 PDF Document Management")

    existing = _list_existing_pdfs()
    if existing:
        with st.sidebar.expander(f"Registered PDFs ({len(existing)})", expanded=False):
            for p in existing:
                st.caption(f"• {p.name}")
    else:
        st.sidebar.caption("No PDFs registered.")

    uploaded = st.sidebar.file_uploader(
        "Upload PDFs (multiple files allowed)",
        type=["pdf"],
        accept_multiple_files=True,
        help="After uploading, click 'Rebuild Index' to update the FAISS database.",
    )

    if not uploaded:
        return

    st.sidebar.caption(f"{len(uploaded)} file(s) selected")

    if not st.sidebar.button("🔄 Rebuild Index", type="primary", width="stretch"):
        return

    st.markdown("---")
    st.subheader("📊 FAISS Index Rebuild")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(percent: int, message: str) -> None:
        progress_bar.progress(percent)
        status_text.info(f"**{percent}%** — {message}")

    try:
        added_files, added_chunks = add_pdfs_to_vector_db(uploaded, on_progress=on_progress)
        progress_bar.progress(100)
        if added_files == 0:
            status_text.warning("⚠️ All files are already registered. No new PDFs were added.")
            st.sidebar.warning("No new PDFs added")
        else:
            status_text.success(
                f"✅ {added_files} PDF(s) and {added_chunks} chunks added to the database."
            )
            st.sidebar.success(f"✅ {added_files} PDF(s) added")
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"❌ Rebuild failed: {e}")
        st.sidebar.error("Index rebuild failed")
