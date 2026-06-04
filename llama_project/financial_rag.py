import streamlit as st
import tempfile
import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Financial RAG", page_icon="🧠", layout="wide")

st.title("🧠 Financial RAG: Parent-Document Retrieval")
st.markdown("**Project 7, Day 3: Implementing Advanced Context Chunking**")

st.info("Upload a financial PDF. We will split it into large 'Parent' blocks, and then subdivide those into smaller 'Child' chunks for precise searching.")

# --- SIDEBAR: UPLOAD ---
st.sidebar.header("📥 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type=["pdf"])

@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- MAIN APP ---
if uploaded_file:
    # 1. Extract Text (From Day 2)
    with st.spinner("Extracting PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()
        os.remove(tmp_path)
    
    st.success(f"Extracted {len(raw_docs)} pages.")

    # 2. Advanced Chunking Setup
    with st.spinner("Building Parent-Document Architecture..."):
        
        # The Core Setup:
        # Parent Splitter: Creates large, context-rich blocks (e.g., a whole page or section)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        # Child Splitter: Creates tiny, highly searchable chunks (e.g., a single sentence or table row)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        # The Vector DB stores the *small* child chunks and their embeddings
        embedder = load_embedder()
        vectorstore = Chroma(
            collection_name="split_parents", 
            embedding_function=embedder
        )
        
        # The Python Dictionary (InMemoryStore) maps a unique ID to the *large* parent chunks
        store = InMemoryStore()

        # The Retriever orchestrates the relationship
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        # 3. Execute the Chunking & Storage
        retriever.add_documents(raw_docs, ids=None)
        st.success("✅ Parent-Document vector database built successfully!")

    # 4. The Test: Let's see what it retrieves
    st.divider()
    st.subheader("🔍 Test the Retrieval Engine")
    st.write("Ask a highly specific question (e.g., a specific revenue number or metric).")
    
    test_query = st.text_input("Enter a test query:")
    
    if test_query:
        # We query the vector database
        retrieved_docs = retriever.invoke(test_query)
        
        if retrieved_docs:
            st.markdown("### Resulting Context Passed to AI")
            st.caption("Notice how the retrieval engine found the specific sentence, but is returning the entire surrounding context block so the AI doesn't lose the bigger picture!")
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"Parent Document Block {i+1}", expanded=True):
                    st.write(doc.page_content)
                    
            # For technical demonstration, let's peek under the hood at the raw Vector DB matches
            st.divider()
            st.subheader("🛠️ Under the Hood: The Child Chunks")
            st.write("These are the actual tiny pieces of text the math matched with your query. Notice how they are mapped to the larger blocks above using a `doc_id`.")
            
            # Direct query to Chroma to see the children
            raw_child_matches = vectorstore.similarity_search(test_query, k=2)
            for child in raw_child_matches:
                st.json({
                    "Child Text": child.page_content,
                    "Metadata": child.metadata
                })
        else:
            st.warning("No matches found.")
else:
    st.write("👈 Please upload a PDF to begin.")