import streamlit as st
import pandas as pd
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Extractor", page_icon="📄", layout="wide")

st.title("📄 Financial Data Extractor")
st.markdown("**Project 7, Day 2: Ingesting Dense Financial Documents**")

st.info("Upload an Earnings Report (PDF) or a Financial Tracker (Excel). This engine will extract the raw text and tables so our local LLaMA can read it.")

# --- SIDEBAR: UPLOAD ---
st.sidebar.header("📥 Document Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload Financial Document", 
    type=["pdf", "xlsx", "csv"]
)

# --- EXTRACTION FUNCTIONS ---
def extract_pdf(file_bytes, filename):
    """Saves the PDF temporarily, uses LangChain to extract text, and cleans up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        # Combine all pages into one large string
        full_text = "\n\n".join([page.page_content for page in pages])
        page_count = len(pages)
    finally:
        os.remove(tmp_path)
        
    return full_text, page_count

def extract_spreadsheet(file_bytes, filename):
    """Reads Excel/CSV files using Pandas and converts them to a Markdown table string."""
    if filename.endswith('.csv'):
        df = pd.read_csv(file_bytes)
    else:
        df = pd.read_excel(file_bytes)
        
    # Convert the DataFrame to a Markdown table so the LLM can easily read the rows/columns
    markdown_table = df.to_markdown(index=False)
    
    # Also capture some basic metadata
    summary = f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
    summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
    
    return summary + markdown_table, 1

# --- MAIN APP ---
if uploaded_file:
    file_name = uploaded_file.name
    file_ext = file_name.split('.')[-1].lower()
    
    with st.spinner(f"Extracting data from {file_name}..."):
        # Process based on file type
        if file_ext == 'pdf':
            extracted_text, count = extract_pdf(uploaded_file.read(), file_name)
            doc_type = "PDF Report"
        else:
            extracted_text, count = extract_spreadsheet(uploaded_file, file_name)
            doc_type = "Financial Spreadsheet"
            
        # Save to session state so we can use it in later days
        st.session_state['financial_text'] = extracted_text
        
    # UI Display
    st.success(f"✅ Extraction Complete! Processed {count} unit(s) from {doc_type}.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Metrics")
        st.metric("Total Characters", f"{len(extracted_text):,}")
        st.metric("Estimated Tokens", f"{int(len(extracted_text) / 4):,}")
        st.caption("Tokens are the 'words' the AI sees. Most local models have a limit of 8,000 tokens per prompt!")
        
    with col2:
        st.subheader("Raw Extracted Output")
        with st.expander("Preview Extracted Text", expanded=True):
            st.text_area("This is exactly what the AI will 'see':", value=extracted_text, height=400)

else:
    st.write("👈 Please upload a PDF or Excel file in the sidebar to begin.")