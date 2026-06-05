import streamlit as st
import tempfile
import os
import requests
import pandas as pd
import json
import re

# --- LANGCHAIN IMPORTS ---
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# --- PAGE CONFIG ---
st.set_page_config(page_title="Offline Financial Analyst", page_icon="📈", layout="wide")
st.title("📈 Offline Financial Analyst & Extractor")
st.markdown("**Project 7, Day 5: Structured KPI Extraction & CSV Export**")

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
    # Temperature 0 is CRITICAL for structured JSON extraction
    return ChatOllama(model=model_name, temperature=0.0)

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
        
    st.info("💡 **Tip:** Structured JSON extraction requires a smarter model. Highly recommend using `llama3` or `phi3` if your hardware allows it.")
    model_choice = st.selectbox("Select Model", ["llama3", "phi3", "qwen2.5:0.5b", "tinyllama"])
    
    st.divider()
    st.header("📥 Document Upload")
    uploaded_file = st.file_uploader("Upload Financial PDF", type=["pdf"])
    
    if st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Upload a document first.")
        else:
            with st.spinner("Processing document securely..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                loader = PyPDFLoader(tmp_path)
                raw_docs = loader.load()
                os.remove(tmp_path)
                
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                
                vectorstore = Chroma(collection_name="financial_kpi_docs", embedding_function=load_embedder())
                store = InMemoryStore()
                
                retriever = ParentDocumentRetriever(
                    vectorstore=vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter,
                )
                
                retriever.add_documents(raw_docs, ids=None)
                st.session_state['retriever'] = retriever
                
                st.session_state.messages = [{"role": "assistant", "content": "Knowledge base compiled offline. Ready for analysis."}]
                st.session_state.chat_history = []
                st.success("✅ Ready!")

# Ensure models are loaded
llm = load_llm(model_choice)

if 'retriever' not in st.session_state:
    st.info("👈 Please upload a PDF and build the knowledge base to begin.")
    st.stop()

retriever = st.session_state['retriever']

def get_context(params):
    docs = retriever.invoke(params["input"])
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# --- TABS FOR DUAL FUNCTIONALITY ---
tab1, tab2 = st.tabs(["💬 Conversational Analyst", "📊 KPI Extraction Engine (JSON/CSV)"])

# --- TAB 1: CONVERSATIONAL ANALYST (DAY 4) ---
with tab1:
    st.subheader("Interactive Q&A")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite financial analyst operating entirely offline. Use the context to answer precisely. Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    chat_chain = RunnablePassthrough.assign(context=get_context) | chat_prompt | llm | StrOutputParser()
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask a question about the document..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing securely..."):
                response = chat_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# --- TAB 2: STRUCTURED EXTRACTION (DAY 5) ---
with tab2:
    st.subheader("Automated KPI Extraction")
    st.write("Force the AI to find specific metrics and output them as structured data.")
    
    # 1. Define the exact structure we want using Pydantic
    class FinancialKPIs(BaseModel):
        company_name: str = Field(description="The name of the company")
        reporting_period: str = Field(description="The quarter or year being reported (e.g., Q4 2023)")
        total_revenue: str = Field(description="Total revenue or sales figure with currency")
        net_income: str = Field(description="Net income or profit figure with currency")
        key_risk_factor: str = Field(description="One major risk factor mentioned in the text")

    parser = JsonOutputParser(pydantic_object=FinancialKPIs)
    
    extraction_prompt = ChatPromptTemplate.from_template(
        "You are a strict data extraction algorithm. Extract the requested financial information from the context below.\n"
        "You must format your output exactly according to the following instructions.\n"
        "If a metric is not found, output 'Not Disclosed'.\n\n"
        "Format Instructions:\n{format_instructions}\n\n"
        "Context:\n{context}\n\n"
        "Query: Extract the core financial KPIs."
    )
    
    extract_chain = (
        RunnablePassthrough.assign(context=get_context)
        | extraction_prompt
        | llm
        | StrOutputParser() # We use StrOutputParser first to catch raw LLM output, then parse it manually for safety with local models
    )
    
    if st.button("🚀 Run Automated Extraction", type="primary"):
        with st.spinner(f"Extracting structured data using {model_choice}..."):
            try:
                # We feed a generic query to trigger the retrieval of broad financial info
                raw_response = extract_chain.invoke({
                    "input": "financial performance revenue net income risk factors",
                    "format_instructions": parser.get_format_instructions()
                })
                
                # Cleanup: Local models sometimes add markdown fences like ```json ... ```
                cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()
                
                # Use Regex to find the JSON block in case the model hallucinated extra text
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                
                if json_match:
                    parsed_json = json.loads(json_match.group())
                    
                    st.success("✅ Data Successfully Structured!")
                    
                    # Convert to Pandas DataFrame for beautiful display and export
                    df_kpi = pd.DataFrame([parsed_json])
                    st.table(df_kpi)
                    
                    # Provide CSV Download
                    csv = df_kpi.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name="extracted_financial_kpis.csv",
                        mime="text/csv",
                    )
                    
                    with st.expander("Under the Hood: Raw JSON Payload"):
                        st.json(parsed_json)
                else:
                    st.error("Model failed to output valid JSON. See raw output below:")
                    st.write(raw_response)
                    
            except Exception as e:
                st.error(f"Extraction Pipeline Error: {e}")
                st.write("Local models sometimes struggle with strict formatting. Try using the `llama3` model.")