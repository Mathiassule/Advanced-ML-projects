import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG: Vector Database", page_icon="🗄️", layout="wide")

st.title("🗄️ RAG Engine: Vector Database & Semantic Search")
st.markdown("**Project 5, Day 2: Turning Text into Math with ChromaDB**")

# --- INITIALIZE EMBEDDING MODEL ---
@st.cache_resource
def load_embedding_model():
    """
    Loads a free, local HuggingFace embedding model.
    all-MiniLM-L6-v2 converts sentences into a 384-dimensional vector.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    with st.spinner("Loading Embedding Model (This might take a minute on first run)..."):
        embedder = load_embedding_model()
except Exception as e:
    st.error(f"Failed to load embedding model: {e}")
    st.stop()

# --- DEFAULT TEXT ---
DEFAULT_TEXT = """
**Acme Corp Employee Handbook - Section 4: Remote Work Policy**
Employees at Acme Corp are permitted to work remotely up to 3 days per week. Core hours are from 10:00 AM to 3:00 PM EST, during which all employees must be available for meetings. A home office stipend of $500 is provided annually to assist with internet and equipment costs.

**Section 5: Paid Time Off (PTO)**
All full-time employees accrue 15 days of PTO per year. PTO rolls over up to a maximum of 5 days into the next calendar year. To request time off, employees must submit a request through the Workday portal at least 14 days in advance for vacations longer than 3 days.

**Section 6: IT Security**
Never share your passwords. All company laptops must be connected to the corporate VPN when accessing internal networks from public Wi-Fi. Report any phishing attempts immediately to IT_Security@acmecorp.com.
"""

# --- APP LAYOUT ---
tab1, tab2 = st.tabs(["🏗️ 1. Build Database", "🔍 2. Semantic Search"])

# --- TAB 1: BUILD DB ---
with tab1:
    st.subheader("Step 1: Chunk & Embed")
    raw_text = st.text_area("Source Document", value=DEFAULT_TEXT.strip(), height=250)
    
    if st.button("🔨 Build Vector Database", type="primary"):
        with st.spinner("Chunking text and calculating embeddings..."):
            
            # 1. Chunking (From Day 1)
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks = splitter.split_text(raw_text)
            
            # Convert string chunks into LangChain Document objects
            documents = [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]
            
            # 2. Vector Database Creation
            # We use an in-memory Chroma database for this demo. 
            # In production, you would save this to disk (persist_directory="./chroma_db")
            vector_db = Chroma.from_documents(
                documents=documents, 
                embedding=embedder
            )
            
            # Save the database instance to session state so we can search it in Tab 2
            st.session_state['vector_db'] = vector_db
            
            st.success(f"✅ Successfully converted {len(chunks)} chunks into vectors and stored them in ChromaDB!")
            
            # Show a visual representation of what an embedding looks like
            st.divider()
            st.write("### 🧠 What did the AI just do?")
            st.write("It took your text and converted it into a mathematical vector. Here is a preview of the embedding for Chunk 0:")
            sample_embedding = embedder.embed_query(chunks[0])
            st.info(f"**Dimensions:** {len(sample_embedding)} numbers long")
            st.code(str(sample_embedding[:10]) + " ... [truncated]", language="python")

# --- TAB 2: SEMANTIC SEARCH ---
with tab2:
    st.subheader("Step 2: Retrieve Information")
    st.write("Ask a question. The database will convert your question into a vector and find the chunks with the mathematically closest meaning.")
    
    query = st.text_input("Search Query:", placeholder="e.g., Do I get money for working at home?")
    
    if st.button("🔍 Search Database"):
        if 'vector_db' not in st.session_state:
            st.warning("⚠️ Please go to Tab 1 and build the database first!")
        elif not query:
            st.warning("⚠️ Please enter a search query.")
        else:
            with st.spinner("Searching vector space..."):
                db = st.session_state['vector_db']
                
                # Perform Similarity Search (Retrieve top 3 closest chunks)
                # We use similarity_search_with_score to see the distance metric
                results = db.similarity_search_with_score(query, k=3)
                
                st.write("### 🎯 Search Results")
                
                for idx, (doc, score) in enumerate(results):
                    # Chroma returns distance (lower score = closer match / higher similarity)
                    with st.expander(f"Result {idx + 1} (Distance Score: {score:.4f})", expanded=True):
                        st.write(f"**Source Text:**")
                        st.success(doc.page_content)
                        st.write(f"*Metadata: {doc.metadata}*")

    st.divider()
    st.markdown("""
    **💡 Try this test:**
    Search for *"vacation"*. Notice how it successfully pulls up the PTO section, even though the word "vacation" is only briefly mentioned at the very end. The AI understands that "PTO", "time off", and "vacation" occupy the same semantic space!
    """)  