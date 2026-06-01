import streamlit as st
from langchain_community.chat_models import ChatOllama
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="Local LLaMA Engine", page_icon="🦙", layout="wide")

st.title("🦙 Local LLaMA: Secure Offline Engine")
st.markdown("**Project 7, Day 1: Establishing the Local API Connection**")

# --- CHECK LOCAL CONNECTION ---
def check_ollama_status():
    """Pings the local Ollama server to see if it is running."""
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

is_connected = check_ollama_status()

# --- SIDEBAR: SETUP INSTRUCTIONS ---
with st.sidebar:
    st.header("⚙️ Server Status")
    
    if is_connected:
        st.success("✅ Connected to Localhost:11434")
    else:
        st.error("❌ Ollama is not running!")
        
    st.divider()
    st.markdown("""
    ### 🛠️ How to start the AI:
    1. Download & Install [Ollama](https://ollama.com/).
    2. Open your computer's Terminal / Command Prompt.
    3. Run one of these commands for a lightweight model:
       ```bash
       # Super light (~400MB)
       ollama run qwen2.5:0.5b
       
       # Very light (~600MB)
       ollama run tinyllama
       
       # Light & Smart (~2.3GB)
       ollama run phi3
       ```
    4. Keep that terminal open in the background!
    """)
    
    # Allow user to switch models if they have others downloaded
    model_choice = st.selectbox("Select Local Model", ["qwen2.5:0.5b", "tinyllama", "phi3", "llama3"])

# --- LOAD LOCAL LLM ---
@st.cache_resource
def load_model(model_name):
    # Upgraded to ChatOllama so the model knows it is in a conversation!
    return ChatOllama(model=model_name)

if is_connected:
    llm = load_model(model_choice)
else:
    st.warning("👈 Please follow the sidebar instructions to start your local AI server.")
    st.stop()

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "System secured. I am running completely offline on your hardware. How can I help you today?"}
    ]

# --- CHAT INTERFACE ---
# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_query := st.chat_input("Ask your local LLaMA model a question..."):
    
    # 1. Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Stream AI Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Generating offline..."):
            
            # Format the UI history into LangChain's strict Chat format
            chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_history.append(("human", msg["content"]))
                else:
                    chat_history.append(("ai", msg["content"]))
            
            # Pass the formatted history to the Chat model
            for chunk in llm.stream(chat_history):
                # Chat models return message objects, so we extract the .content
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
                
            # Final output without the blinking cursor
            response_placeholder.markdown(full_response)
            
    # 3. Save to memory
    st.session_state.messages.append({"role": "assistant", "content": full_response})