"""A small Streamlit interface for asking questions about uploaded PDFs."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")
# Load configuration from the project's .env file, regardless of where Streamlit was launched.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the context does not contain the answer, say that you do not know.
Keep the answer clear, and explain completely using your orchestration way, and concise.

<context>
{context}
</context>

Question: {input}"""
)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model once per Streamlit server."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store(pdf_bytes: bytes, filename: str) -> tuple[FAISS, int]:
    """Load a PDF from the upload, split it, and index its chunks."""
    suffix = Path(filename).suffix or ".pdf"
    temp_path = ""

    try:
        # PyPDFLoader needs a file path, so save the uploaded bytes briefly before reading them.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        pages = PyPDFLoader(temp_path).load()
    finally:
        # Always remove the temporary upload, including when the PDF cannot be read.
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    # Smaller overlapping chunks preserve context without making prompts too large.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("No readable text was found in this PDF.")

    return FAISS.from_documents(chunks, get_embeddings()), len(chunks)


def get_answer(vector_store: FAISS, question: str) -> tuple[str, list]:
    """Retrieve relevant chunks and generate an answer with Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file and restart Streamlit.")

    # Gemini answers from the passages selected by the retriever.
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=api_key)
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        document_chain,
    )
    result = retrieval_chain.invoke({"input": question})
    return result["answer"], result["context"]


st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF, then ask questions grounded in its contents.")

with st.sidebar:
    st.header("Document")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    st.caption("Your document is indexed only for this running app session.")

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.getvalue()
    # The file hash avoids rebuilding the index whenever Streamlit reruns the page.
    document_id = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("document_id") != document_id:
        try:
            with st.spinner("Reading and indexing the PDF..."):
                vector_store, chunk_count = build_vector_store(pdf_bytes, uploaded_pdf.name)
            st.session_state.document_id = document_id
            st.session_state.vector_store = vector_store
            st.session_state.chunk_count = chunk_count
            # A new document starts a fresh, relevant conversation.
            st.session_state.messages = []
        except Exception as error:
            st.error(f"Could not process this PDF: {error}")

    if "vector_store" in st.session_state:
        st.success(f"Ready — indexed {st.session_state.chunk_count} text chunks.")

        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about this PDF")
        if question:
            # Keep each turn visible after Streamlit reruns the app.
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching the document and writing an answer..."):
                        answer, source_docs = get_answer(st.session_state.vector_store, question)
                    st.markdown(answer)
                    with st.expander("Retrieved source passages"):
                        # Let users verify the answer against the passages used to produce it.
                        for index, document in enumerate(source_docs, start=1):
                            page = document.metadata.get("page")
                            page_label = f"Page {page + 1}" if isinstance(page, int) else "Source passage"
                            st.markdown(f"**{index}. {page_label}**")
                            st.write(document.page_content)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as error:
                    st.error(str(error))
else:
    st.info("Upload a PDF from the sidebar to begin.")
