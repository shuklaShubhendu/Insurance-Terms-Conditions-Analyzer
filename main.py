import streamlit as st
import os
import tempfile
from pypdf import PdfReader
from docx import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set Streamlit layout
st.set_page_config(page_title="Insurance Analyzer", layout="wide")

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY in your Streamlit Secrets.")
    st.stop()

# Initialize Langchain LLM and embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# --- Document Reading Helpers ---
def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def read_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def process_document(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return Chroma.from_texts(texts=chunks, embedding=embeddings, collection_name="insurance_doc")

def create_qa_chain(vector_store):
    prompt_template = """
You are an expert insurance document analyst. Use the following context to answer the question.

Context: {context}

Question: {question}

Provide a concise and accurate answer.
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )

# --- Streamlit UI ---
st.title("🛡️ Insurance Terms & Conditions Analyzer (RAG)")
st.markdown("Analyze your insurance document using AI-powered retrieval and summarization.")

uploaded_file = st.file_uploader("📁 Upload your insurance document (PDF, DOCX, DOC)", type=["pdf", "doc", "docx"])

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.qa_chain = None
    st.session_state.analysis_done = False
    st.session_state.chat_history = []

# Process file
if uploaded_file and not st.session_state.analysis_done:
    ext = uploaded_file.name.split(".")[-1].lower()
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File size exceeds 10 MB limit.")
        st.stop()

    with st.spinner("🔍 Processing document..."):
        try:
            if ext == "pdf":
                content = read_pdf(uploaded_file)
            elif ext in ["docx", "doc"]:
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + ext) as tmp:
                    tmp.write(uploaded_file.read())
                    content = read_docx(tmp.name)
                    os.remove(tmp.name)
            else:
                st.error("Unsupported file format.")
                st.stop()

            vs = process_document(content)
            qa = create_qa_chain(vs)

            st.session_state.vector_store = vs
            st.session_state.qa_chain = qa
            st.session_state.analysis_done = True

            # Initial analysis
            with st.spinner("Generating initial insights..."):
                st.session_state.summary = qa.run("Summarize this insurance document.")
                st.session_state.red_flags = qa.run("List any red flags or unclear terms.")
                st.session_state.important_points = qa.run("Highlight important points and potential hidden charges.")

        except Exception as e:
            st.error(f"Error: {e}")

# Show analysis
if st.session_state.analysis_done:
    st.markdown("---")
    st.subheader("📄 Document Analysis")
    st.markdown(f"**📝 Summary:**\n{st.session_state.summary}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🚩 **Red Flags & Vague Terms**")
        st.markdown(st.session_state.red_flags)
    with col2:
        st.markdown("🔎 **Important Points & Hidden Charges**")
        st.markdown(st.session_state.important_points)

    # Chat interface
    st.markdown("---")
    st.subheader("💬 Chat with Your Document")
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "assistant", "content": "Hi there! Ask me anything about your document."})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(prompt)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
