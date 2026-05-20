import streamlit as st
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG: Full Pipeline", page_icon="🤖", layout="wide")

# --- IMPORTS ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- APP STARTS HERE ---
st.title("🤖 RAG Engine: Full Pipeline")
st.markdown("**Project 5, Day 3: Connecting the Vector DB to an LLM**")

# --- SIDEBAR: API KEY ---
st.sidebar.header("🔑 Authentication")
google_api_key = st.sidebar.text_input("Google API Key", type="password", help="Get this from Google AI Studio.")

if not google_api_key:
    st.warning("👈 Please enter your Google API Key in the sidebar to continue.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

# --- LOAD MODELS ---
@st.cache_resource
def load_embedder():
    # We will keep using the local HuggingFace embeddings for the Vector DB to save costs!
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    # Using Google's incredibly fast and stable Gemini 2.5 Flash model!
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=512
    )

with st.spinner("Initializing AI Models..."):
    embedder = load_embedder()
    llm = load_llm()

# --- DEFAULT TEXT ---
DEFAULT_TEXT = """
**Acme Corp Employee Handbook - Section 4: Remote Work Policy**
Employees at Acme Corp are permitted to work remotely up to 3 days per week. Core hours are from 10:00 AM to 3:00 PM EST, during which all employees must be available for meetings. A home office stipend of $500 is provided annually to assist with internet and equipment costs.

**Section 5: Paid Time Off (PTO)**
All full-time employees accrue 15 days of PTO per year. PTO rolls over up to a maximum of 5 days into the next calendar year. To request time off, employees must submit a request through the Workday portal at least 14 days in advance for vacations longer than 3 days.
"""

# --- THE RAG PROMPT TEMPLATE ---
system_prompt = (
    "You are a helpful, professional assistant for Acme Corp. "
    "Use the following pieces of retrieved context to answer the user's question. "
    "If you don't know the answer, just say that you don't know. Do NOT make up information.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- APP LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Ingest Data")
    raw_text = st.text_area("Source Document", value=DEFAULT_TEXT.strip(), height=200)
    
    if st.button("🔨 Build Knowledge Base", type="primary"):
        with st.spinner("Building Vector Database..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks = splitter.split_text(raw_text)
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Create Chroma DB
            vector_db = Chroma.from_documents(documents=documents, embedding=embedder)
            st.session_state['retriever'] = vector_db.as_retriever(search_kwargs={"k": 2})
            st.success("✅ Knowledge base built and ready!")

with col2:
    st.subheader("2. Ask the AI")
    user_query = st.text_input("Ask a question about the document:")
    
    if st.button("💬 Generate Answer"):
        if 'retriever' not in st.session_state:
            st.error("⚠️ Please build the knowledge base first!")
        elif not user_query:
            st.warning("⚠️ Please ask a question.")
        else:
            with st.spinner("Thinking..."):
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                # 1. Setup Parallel Retrieval (Fetch docs + pass user input)
                setup_and_retrieval = RunnableParallel(
                    {"context": st.session_state['retriever'], "input": RunnablePassthrough()}
                )
                
                # 2. Define the Generation Step
                generation = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # 3. Combine into Modern LCEL Chain
                rag_chain = setup_and_retrieval.assign(answer=generation)
                
                # 4. Execute the pipeline with graceful Error Handling
                try:
                    response = rag_chain.invoke(user_query)
                    
                    st.markdown("### 🤖 Answer:")
                    st.info(response["answer"])
                    
                    with st.expander("🔍 See Retrieved Chunks (What the AI read)"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**Chunk {i+1}:** {doc.page_content}")
                            
                except Exception as e:
                    error_msg = str(e)
                    if "11001" in error_msg or "getaddrinfo" in error_msg:
                        st.error("🚨 Network Connection Error: Your computer cannot reach the Google servers.")
                        st.warning("This is an internet or DNS issue. Please check your Wi-Fi, turn off any strict VPNs, or verify your firewall isn't blocking Python from accessing the internet.")
                    else:
                        st.error("🚨 An error occurred during generation.")
                        st.code(error_msg)