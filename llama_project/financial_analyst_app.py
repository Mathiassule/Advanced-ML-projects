import streamlit as st
import tempfile
import os
import requests
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
# Using the correct package for the retriever!
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Local Financial Analyst", page_icon="📈", layout="wide")
st.title("📈 Offline Financial Analyst")
st.markdown("**Project 7, Day 4: Full End-to-End Integration**")

# --- SERVER CHECK ---
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

is_connected = check_ollama_status()

# --- LOAD MODELS ---
@st.cache_resource
def load_llm(model_name):
    return ChatOllama(model=model_name, temperature=0.1)

@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- SIDEBAR: SETUP & INGESTION ---
with st.sidebar:
    st.header("⚙️ System Status")
    if is_connected:
        st.success("✅ Localhost:11434 Connected")
    else:
        st.error("❌ Ollama Offline")
        st.stop()
        
    model_choice = st.selectbox("Select Model", ["qwen2.5:0.5b", "tinyllama", "phi3", "llama3"])
    
    st.divider()
    st.header("📥 Document Upload")
    uploaded_file = st.file_uploader("Upload Financial PDF", type=["pdf"])
    
    if st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Upload a document first.")
        else:
            with st.spinner("Processing document..."):
                # 1. Parse
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                loader = PyPDFLoader(tmp_path)
                raw_docs = loader.load()
                os.remove(tmp_path)
                
                # 2. Advanced Chunking
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                
                vectorstore = Chroma(collection_name="financial_docs", embedding_function=load_embedder())
                store = InMemoryStore()
                
                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )
                
                retriever.add_documents(raw_docs, ids=None)
                st.session_state['retriever'] = retriever
                
                # Reset memory for new document
                st.session_state.messages = [{"role": "assistant", "content": "Knowledge base compiled offline. What financial metrics would you like to analyze?"}]
                st.session_state.chat_history = []
                st.success("✅ Ready!")

# Ensure models are loaded
llm = load_llm(model_choice)

if 'retriever' not in st.session_state:
    st.info("👈 Please upload a PDF and build the knowledge base to begin.")
    st.stop()

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Knowledge base compiled offline. What financial metrics would you like to analyze?"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- LCEL RAG PIPELINE ---
system_prompt = (
    "You are an elite financial analyst operating entirely offline. "
    "Use the following pieces of context to answer the user's question. "
    "Be precise, reference specific numbers where possible, and do not hallucinate.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

retriever = st.session_state['retriever']

def get_context(params):
    docs = retriever.invoke(params["input"])
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(context=get_context)
    | prompt
    | llm
    | StrOutputParser()
)

# --- CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Analyze the document..."):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing securely..."):
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))