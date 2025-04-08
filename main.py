import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import tempfile
import os

st.set_page_config(page_title="Insurance Analyzer", layout="wide")

st.title("🛡️ Insurance Terms & Conditions Analyzer (RAG)")
st.markdown("Analyze your insurance document using AI-powered retrieval and summarization.")

uploaded_file = st.file_uploader(
    "📁 Upload your insurance document (PDF, DOCX, DOC)", 
    type=["pdf", "docx", "doc"]
)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

def load_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif suffix in [".docx", ".doc"]:
        loader = Docx2txtLoader(tmp_file_path)
    else:
        st.error("Unsupported file format.")
        return []

    documents = loader.load()
    os.remove(tmp_file_path)
    return documents

def process_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(documents=chunks, embedding=embeddings)

if uploaded_file:
    st.info(f"📄 Uploaded: {uploaded_file.name}")
    
    with st.spinner("Processing document and building knowledge base..."):
        documents = load_file(uploaded_file)
        if documents:
            vectorstore = process_documents(documents)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.success("✅ Document processed. Ask me anything from it!")

            user_question = st.text_input("💬 Ask a question about your document:")
            if user_question:
                with st.spinner("Thinking..."):
                    response = qa_chain.run(user_question)
                    st.markdown(f"**Answer:** {response}")
        else:
            st.error("Failed to load document.")
