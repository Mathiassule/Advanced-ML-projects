import streamlit as st
import os
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="Enterprise Document AI", page_icon="📄", layout="wide")

# --- IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- TITLE & UI HEADER ---
st.title("📄 Enterprise Document AI")
st.markdown("**Project 5, Day 5: Production RAG System with PDF Ingestion**")

# --- LOAD MODEL FUNCTIONS ---
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2, # Slightly lower temperature for stricter fact-based answers
        max_output_tokens=1024
    )

# --- SIDEBAR: CONFIG & INGESTION ---
with st.sidebar:
    st.header("🔑 Authentication")
    google_api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    
    st.header("📂 Document Upload")
    st.info("Upload any PDF document to chat with it.")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        if not google_api_key:
            st.error("Please enter your Google API Key first!")
        elif not uploaded_file:
            st.error("Please upload a PDF document.")
        else:
            with st.spinner("Analyzing document..."):
                os.environ["GOOGLE_API_KEY"] = google_api_key
                
                # 1. Save uploaded file temporarily for PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # 2. Extract Text
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                raw_text = "\n".join([page.page_content for page in pages])
                os.remove(tmp_file_path) # Clean up temp file
                
                # 3. Chunking
                embedder = load_embedder()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = splitter.split_text(raw_text)
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                # 4. Vector Database Build
                vector_db = Chroma.from_documents(documents=documents, embedding=embedder)
                st.session_state['retriever'] = vector_db.as_retriever(search_kwargs={"k": 4}) # Fetch top 4 chunks
                
                # 5. Reset Memory for the new document
                st.session_state.messages = [{"role": "assistant", "content": f"I have successfully read your document ({len(pages)} pages). What would you like to know?"}]
                st.session_state.chat_history = []
                
                st.success("✅ Document processed and memorized!")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        if 'messages' in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. What else would you like to ask about the document?"}]
            st.session_state.chat_history = []
            st.rerun()

# --- APP STARTUP CHECKS ---
if not google_api_key:
    st.warning("👈 Please enter your Google API Key in the sidebar.")
    st.stop()
if 'retriever' not in st.session_state:
    st.info("👈 Please upload a PDF and click 'Process Document' to begin.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
llm = load_llm()

# --- STATE MANAGEMENT (MEMORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am your Document Assistant. Ask me anything about the text you provided!"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- LCEL RAG PIPELINE ---
system_prompt = (
    "You are an expert document analysis AI. "
    "Use the following pieces of retrieved context to answer the user's question accurately. "
    "If the answer is not explicitly contained in the context, politely state that you cannot find the answer in the provided document. Do NOT hallucinate.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Grab the retriever safely in the main thread
retriever = st.session_state['retriever']

def get_retrieved_context(params):
    query = params["input"]
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(context=get_retrieved_context)
    | prompt
    | llm
    | StrOutputParser()
)

# --- CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about your PDF..."):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Scanning document..."):
            
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))