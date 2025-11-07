import uuid
import os
import base64
import streamlit as st
from query_engine import QueryEngine
from utils.vector_utils import VectorStore
from utils.docs_utils import extract_pdf_text, extract_txt_text, extract_csv_text, extract_docx_text, chunk_text
from utils.uploads_utils import add_document, get_documents, delete_document
from utils.logging_utils import log_error

st.set_page_config(page_title="Local GPT RAG Assistant")

store = VectorStore()
engine = QueryEngine()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "already_processed" not in st.session_state:
    st.session_state.already_processed = False

st.title("📄 Local GPT RAG Assistant")
st.subheader("Upload → Index → Ask Questions")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf","docx","txt","csv"],
    accept_multiple_files=True,
    key="file_uploader"
)
if uploaded_files and not st.session_state.already_processed:
    with st.spinner("Indexing uploaded documents..."):
        for file in uploaded_files:
            try:
                raw = file.getvalue()
                ext = file.name.lower()

                save_path = os.path.join(UPLOAD_DIR, file.name)
                with open(save_path, "wb") as f:
                    f.write(raw)

                if ext.endswith(".pdf"):
                    text = extract_pdf_text(raw)
                elif ext.endswith(".docx"):
                    text = extract_docx_text(raw)
                elif ext.endswith(".csv"):
                    text = extract_csv_text(raw)
                else:
                    text = extract_txt_text(raw)

                chunks = chunk_text(text)
                ids = [str(uuid.uuid4()) for _ in chunks]
                metas = [{"file": file.name, "chunk": i} for i in range(len(chunks))]
                store.upsert(ids, chunks, metas)

                add_document(file.name, save_path)
            except Exception as e:
                log_error(f"Failed while indexing {file.name}: {str(e)}")

    st.session_state.already_processed = True
    st.rerun()

docs = get_documents()
if docs:
    st.write("#### Indexed Documents History")

    for doc in docs:
        col1, col2, col3, col4 = st.columns([4, 3, 2, 1])
        col1.write(doc["name"])
        col2.write(doc["created_at"])

        with open(doc["path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            col3.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="{doc["name"]}">Download</a>',unsafe_allow_html=True)
            
        delete = col4.button("🗑️", key=f"delete-{doc['name']}")
        if delete:
            delete_document(doc["name"])
            store.delete_by_filename(doc["name"])
            os.remove(doc["path"])
            st.rerun()

st.write("---")

query = st.text_input("## Ask a Query")

if st.button("Search"):
    if not query:
        st.warning("Enter a question.")
    else:
        try:
            res = engine.ask(query, top_k=5, threshold=0.20)
            st.subheader("Answer")
            st.write(res["answer"])

            if res["sources"]:
                st.subheader("Sources")
            for s in res["sources"]:
                chunk_id = s["id"]
                similarity = s["similarity"]

                # get chunk text from vector DB (already stored)
                chunk_record = store.col.get(ids=[chunk_id])
                chunk_text = chunk_record["documents"][0] if chunk_record and chunk_record["documents"] else ""

                st.write(f"🔹 **Chunk ID:** `{chunk_id}` — similarity: **{similarity:.2f}**")
                with st.expander("View Chunk Content"):
                    st.write(chunk_text)
        except Exception as e:
            log_error(f"Query failed: {str(e)}")
            st.error("Something went wrong while processing the query.")
