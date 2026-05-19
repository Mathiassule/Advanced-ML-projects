import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG: Document Processing", page_icon="✂️", layout="wide")

st.title("✂️ RAG Engine: Document Chunking")
st.markdown("**Project 5, Day 1: Preparing Data for Vector Databases**")

# --- DEFAULT TEXT (Fallback if no upload) ---
DEFAULT_TEXT = """
**Acme Corp Employee Handbook - Section 4: Remote Work Policy**
Employees at Acme Corp are permitted to work remotely up to 3 days per week. Core hours are from 10:00 AM to 3:00 PM EST, during which all employees must be available for meetings. A home office stipend of $500 is provided annually to assist with internet and equipment costs.

**Section 5: Paid Time Off (PTO)**
All full-time employees accrue 15 days of PTO per year. PTO rolls over up to a maximum of 5 days into the next calendar year. To request time off, employees must submit a request through the Workday portal at least 14 days in advance for vacations longer than 3 days.

**Section 6: IT Security**
Never share your passwords. All company laptops must be connected to the corporate VPN when accessing internal networks from public Wi-Fi. Report any phishing attempts immediately to IT_Security@acmecorp.com.
"""

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Chunking Parameters")
st.sidebar.info("How should we slice the document? LLMs need small, manageable pieces.")

chunk_size = st.sidebar.slider("Chunk Size (Characters)", 50, 1000, 200, step=50, 
                               help="The maximum size of each text piece.")
chunk_overlap = st.sidebar.slider("Chunk Overlap (Characters)", 0, 200, 50, step=10, 
                                  help="Overlap prevents sentences from being cut in half, preserving context.")

# Initialize the Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", " ", ""] # It tries to split at paragraphs first, then lines, then spaces
)

# --- APP LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Ingest Document")
    input_method = st.radio("Choose Input Method:", ["Use Sample Text", "Upload PDF Document"])
    
    raw_text = ""
    
    if input_method == "Use Sample Text":
        raw_text = st.text_area("Sample Document Text", value=DEFAULT_TEXT.strip(), height=300)
        
    elif input_method == "Upload PDF Document":
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file:
            # We must save the uploaded file temporarily so LangChain's PDF loader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            with st.spinner("Extracting text from PDF..."):
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                # Combine all pages into one giant string
                raw_text = "\n".join([page.page_content for page in pages])
                st.success(f"Successfully extracted {len(pages)} pages!")
            
            os.remove(tmp_file_path)

with col2:
    st.subheader("2. Analyze Chunks")
    
    if raw_text:
        if st.button("✂️ Process & Split Document", type="primary"):
            with st.spinner("Applying Recursive Character Splitting..."):
                
                # Split the text
                chunks = text_splitter.split_text(raw_text)
                
                st.success(f"Document split into {len(chunks)} chunks!")
                
                # Create a DataFrame to beautifully display the chunks
                chunk_data = []
                for i, chunk in enumerate(chunks):
                    chunk_data.append({
                        "Chunk ID": f"Chunk_{i+1}",
                        "Length": len(chunk),
                        "Text Preview": chunk
                    })
                    
                df_chunks = pd.DataFrame(chunk_data)
                st.dataframe(df_chunks, use_container_width=True)
                
                st.divider()
                st.markdown("### 🔍 Inspecting the Overlap")
                st.info("Notice how the end of one chunk is repeated at the beginning of the next. This overlap ensures the AI doesn't lose the meaning of a sentence that got cut in half!")
                
                if len(chunks) > 1:
                    st.write("**End of Chunk 1:**")
                    st.code(chunks[0][-100:])
                    st.write("**Start of Chunk 2:**")
                    st.code(chunks[1][:100])
    else:
        st.warning("Please provide some text or upload a PDF to begin.") 