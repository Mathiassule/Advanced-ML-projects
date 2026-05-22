import streamlit as st
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --- IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- TITLE ---
st.title("🤖 RAG Engine: Conversational Agent")
st.markdown("**Project 5, Day 4: Adding Chat History & Memory**")

# --- LOAD MODEL FUNCTIONS ---
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=512
    )

# --- SIDEBAR: CONFIG & KNOWLEDGE BASE ---
with st.sidebar:
    st.header("🔑 Authentication")
    google_api_key = st.text_input("Google API Key", type="password", help="Get this from Google AI Studio.")
    
    st.divider()
    
    st.header("📚 Knowledge Base")
    st.info("Ingest documents for the AI to read.")
    
    DEFAULT_TEXT = """
    **Paste text here
    
    """
    
    raw_text = st.text_area("Source Document", value=DEFAULT_TEXT.strip(), height=250)
    
    if st.button("🔨 Build Vector Database", type="primary", use_container_width=True):
        if not google_api_key:
            st.error("Please enter your Google API Key first!")
        else:
            with st.spinner("Embedding document..."):
                os.environ["GOOGLE_API_KEY"] = google_api_key
                embedder = load_embedder()
                splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
                chunks = splitter.split_text(raw_text)
                documents = [Document(page_content=chunk) for chunk in chunks]
                
                vector_db = Chroma.from_documents(documents=documents, embedding=embedder)
                st.session_state['retriever'] = vector_db.as_retriever(search_kwargs={"k": 2})
                
                # Clear chat history when a new database is built so it doesn't remember old docs
                st.session_state.messages = [{"role": "assistant", "content": "Hi! I am your Document Assistant. Ask me anything about the text you provided!"}]
                st.session_state.chat_history = []
                
                st.success("✅ Knowledge base active!")

# Stop the app if the key is missing or DB isn't built
if not google_api_key:
    st.warning("👈 Please enter your Google API Key in the sidebar.")
    st.stop()
if 'retriever' not in st.session_state:
    st.info("👈 Please click 'Build Vector Database' in the sidebar to start chatting.")
    st.stop()

# Ensure API key is set in environment for this run
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize the LLM
llm = load_llm()
 
# --- STATE MANAGEMENT (MEMORY) ---
if "messages" not in st.session_state:
    # Generic greeting instead of Acme Corp
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am your Document Assistant. Ask me anything about the text you provided!"}]
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- LCEL RAG PIPELINE ---
# 1. The Prompt (Generic assistant instead of Acme Corp)
system_prompt = (
    "You are a helpful, professional AI assistant. "
    "Use the following pieces of retrieved context to answer the user's question. "
    "If the answer is not in the context, say 'I don't know based on the provided document.'\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Grab the retriever in the main thread
retriever = st.session_state['retriever']

# 2. Context Retrieval Helper
def get_retrieved_context(params):
    query = params["input"]
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

# 3. The Chain
rag_chain = (
    RunnablePassthrough.assign(context=get_retrieved_context)
    | prompt
    | llm
    | StrOutputParser()
)

# --- CHAT INTERFACE ---
# Display historical messages in the UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_query := st.chat_input("Ask a question..."):
    
    # 1. Display and save user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Pass the input AND the history to the LCEL chain
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            
            st.markdown(response)
            
    # 3. Save to UI History
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 4. Save to LangChain Memory History
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))