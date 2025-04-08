import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from pypdf import PdfReader
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(layout="wide")
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize models and embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# --- Helper Functions ---
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    document = Document(file)
    text = "\n".join([paragraph.text for paragraph in document.paragraphs])
    return text

def process_document(file_content):
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(file_content)
    
    # Create vector store
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="insurance_doc"
    )
    return vector_store

def create_qa_chain(vector_store):
    # Define prompt template
    prompt_template = """You are an expert insurance document analyst. Use the following context to answer questions about the insurance document:
    
    Context: {context}
    
    Question: {question}
    
    Answer concisely and accurately based on the provided context."""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Streamlit UI ---
st.title("🛡️ Insurance Terms & Conditions Analyzer (RAG)")
st.markdown("Analyze your insurance policy using advanced RAG technology.")

uploaded_file = st.file_uploader("📁 Upload your insurance document (PDF, DOC, DOCX)", type=["pdf", "doc", "docx"])

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# Process uploaded file
if uploaded_file is not None and not st.session_state["analysis_done"]:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    max_file_size_mb = 10
    
    if uploaded_file.size > max_file_size_mb * 1024 * 1024:
        st.error(f"File size exceeds the limit of {max_file_size_mb} MB.")
    else:
        with st.spinner("🔍 Processing document..."):
            try:
                if file_extension == "pdf":
                    file_content = read_pdf(uploaded_file)
                elif file_extension == "docx":
                    file_content = read_docx(uploaded_file)
                elif file_extension == "doc":
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        file_content = read_docx(tmp_file.name)
                        os.remove(tmp_file.name)
                else:
                    st.error("Unsupported file type.")
                    st.stop()

                # Process with RAG
                st.session_state["vector_store"] = process_document(file_content)
                st.session_state["qa_chain"] = create_qa_chain(st.session_state["vector_store"])
                st.session_state["analysis_done"] = True
                
                # Initial analysis
                summary = st.session_state["qa_chain"].run("Provide a brief summary of the insurance document")
                red_flags = st.session_state["qa_chain"].run("Identify potential red flags and vague terms")
                important_points = st.session_state["qa_chain"].run("Highlight important points and potential hidden charges")
                
                st.session_state["summary"] = summary
                st.session_state["red_flags"] = red_flags
                st.session_state["important_points"] = important_points
                
            except Exception as e:
                st.error(f"Error processing document: {e}")

# Display analysis
if st.session_state["analysis_done"]:
    st.markdown("---")
    st.subheader("📄 Document Analysis:")
    
    st.markdown(f"**Summary:**\n{st.session_state['summary']}")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚩 Potential Red Flags:**")
        st.markdown(st.session_state['red_flags'])
    with col2:
        st.markdown("**🔑 Important Points:**")
        st.markdown(st.session_state['important_points'])

    # Chat interface
    st.markdown("---")
    st.subheader("🗣️ Chat with the Document:")
    
    if not st.session_state["chat_history"]:
        st.session_state["chat_history"] = [{"role": "assistant", "content": "Hello! Ask me anything about your insurance document."}]
    
    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your questions here"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state["qa_chain"].run(prompt)
                st.markdown(response)
                st.session_state["chat_history"].append({"role": "assistant", "content": response})

# Cleanup ChromaDB when done (optional)
def cleanup():
    if st.session_state["vector_store"] is not None:
        st.session_state["vector_store"].delete_collection()

import atexit
atexit.register(cleanup)