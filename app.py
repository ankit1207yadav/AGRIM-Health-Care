import streamlit as st
import os
import sys
import asyncio
import tempfile
import contextlib
from io import StringIO

# Page config for high-end aesthetics
st.set_page_config(
    page_title="AGRIM Health Care: Clinical Graph-RAG Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Medical Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(90, 92, 234, 0.07) 0%, rgba(255, 255, 255, 0) 90%), 
                    radial-gradient(circle at 90% 80%, rgba(0, 198, 255, 0.05) 0%, rgba(255, 255, 255, 0) 90%);
        background-attachment: fixed;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Custom status boxes */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    
    .action-badge {
        display: inline-block;
        background: #eef2f7;
        color: #4e4376;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #dcdfe6;
    }
</style>
""", unsafe_allow_html=True)

# Context manager to capture printing from the agent logic
@contextlib.contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout

# Sidebar for configuration variables
st.sidebar.markdown("## ⚙️ Configuration Setup")
st.sidebar.write("Configure environment keys below to interact with the LLMs and Neo4j Database.")

openai_key = st.sidebar.text_input(
    "OpenAI API Key", 
    value=os.environ.get("OPENAI_API_KEY", ""), 
    type="password",
    help="Required to generate embedding and execute RAG agents."
)

neo4j_uri = st.sidebar.text_input(
    "Neo4j Connection URI", 
    value=os.environ.get("NEO4J_URI", "neo4j+s://50ce1299.databases.neo4j.io")
)

neo4j_username = st.sidebar.text_input(
    "Neo4j Username", 
    value=os.environ.get("NEO4J_USERNAME", "neo4j")
)

neo4j_password = st.sidebar.text_input(
    "Neo4j Password", 
    value=os.environ.get("NEO4J_PASSWORD", "yLNiXxofzR4GJ3GjUUG6GdjDJ5oGAY4vvCSJY45DOKw"),
    type="password"
)

# Apply configuration to environment
if st.sidebar.button("💾 Apply Configuration", use_container_width=True):
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    st.sidebar.success("Configuration applied to environment!")

# Check credentials before allowing execution
credentials_ok = bool(openai_key and neo4j_uri and neo4j_username and neo4j_password)

if not credentials_ok:
    st.sidebar.warning("⚠️ Please provide all API credentials to enable Ingestion & Chatbot tabs.")

# Main app title
st.markdown("<h1 class='main-header'>🧬 AGRIM Health Care</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Clinical Graph-RAG Assistant (LangGraph & Neo4j Knowledge Networks)</p>", unsafe_allow_html=True)

# Sidebar current credentials overview
if credentials_ok:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🟢 Environment Status")
    st.sidebar.write(f"**Neo4j Endpoint:** `{neo4j_uri.split('://')[-1]}`")
    st.sidebar.write(f"**OpenAI API Status:** `Loaded` ✅")

# Setting up tabs
tab1, tab2 = st.tabs(["📄 PDF Ingest & Graph Generator", "💬 Agentic Graph-RAG Chat"])

# --- TAB 1: INGESTION ---
with tab1:
    st.markdown("### 📤 Parse PDF and Generate Knowledge Graph")
    st.write("Upload a clinical document or patient lab report. The system splits the text, extracts key entities and atomic propositions via GPT-4o, and links them sequentially and relationally inside the Neo4j database.")
    
    uploaded_file = st.file_uploader("Select Clinical PDF file...", type=["pdf"])
    
    if uploaded_file is not None:
        if not credentials_ok:
            st.error("Please configure and apply your API credentials in the sidebar first!")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📄 **File Selected:** `{uploaded_file.name}` ({len(uploaded_file.getvalue())} bytes)")
                chunk_size = st.number_input("Chunk Size (Tokens)", min_value=100, max_value=2000, value=500)
                chunk_overlap = st.number_input("Chunk Overlap (Tokens)", min_value=10, max_value=500, value=100)
            
            with col2:
                st.write("")
                st.write("")
                if st.button("🔨 Build Knowledge Graph", use_container_width=True):
                    with st.spinner("Processing PDF document and injecting relationships to Neo4j... (This may take a moment)"):
                        # Save file to temp folder
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_filepath = tmp_file.name
                        
                        try:
                            # Dynamically import KnowledgeGraph to reflect environment variables
                            import KnowledgeGraph
                            
                            # Capture stdout logs during the async extraction process
                            with capture_stdout() as captured:
                                # Run async main function with filepath
                                result = asyncio.run(KnowledgeGraph.main(temp_filepath))
                            
                            os.remove(temp_filepath)
                            
                            if "Success" in result:
                                st.success("🎉 Knowledge Graph generated successfully in Neo4j!")
                                st.markdown("#### Ingestion Process Trace:")
                                st.text_area("Console Output Logs", value=captured.getvalue(), height=200)
                            else:
                                st.error(f"Failed to generate graph: {result}")
                        except Exception as e:
                            st.error(f"Error during graph creation process: {e}")
                            st.exception(e)

# --- TAB 2: CHATBOT ---
with tab2:
    st.markdown("### 💬 Ask Clinical RAG Agent")
    st.write("Type a question regarding the uploaded medical history. An agentic state graph will formulate a search plan and traverse the Neo4j graph nodes to discover the answer.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "actions" in message:
                st.markdown("**Trajectory Actions:**")
                for action in message["actions"]:
                    st.markdown(f"<span class='action-badge'>{action}</span>", unsafe_allow_html=True)
            if "analysis" in message and message["analysis"]:
                with st.expander("Show Reasoning Synthesis"):
                    st.markdown(message["analysis"])
                    
    # Chat input
    if query := st.chat_input("Enter your health document question here..."):
        if not credentials_ok:
            st.error("Please configure and apply your API credentials in the sidebar first!")
        else:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
                
            with st.chat_message("assistant"):
                with st.spinner("🤖 Agentic state-machine traversing Neo4j knowledge network..."):
                    try:
                        # Ensure environment variables are loaded
                        os.environ["OPENAI_API_KEY"] = openai_key
                        os.environ["NEO4J_URI"] = neo4j_uri
                        os.environ["NEO4J_USERNAME"] = neo4j_username
                        os.environ["NEO4J_PASSWORD"] = neo4j_password
                        
                        # Dynamically import Chatbot to reflect env changes
                        import Chatbot
                        
                        # Run query and capture trace logs
                        with capture_stdout() as captured:
                            response = Chatbot.get_response(query)
                            
                        answer = response.get("answer", "No answer generated.")
                        analysis = response.get("analysis", "")
                        actions = response.get("previous_actions", [])
                        
                        st.markdown(answer)
                        
                        # Show trajectory
                        st.markdown("**Trajectory Actions:**")
                        for action in actions:
                            st.markdown(f"<span class='action-badge'>{action}</span>", unsafe_allow_html=True)
                            
                        # Show reasoning synthesis
                        if analysis:
                            with st.expander("Show Reasoning Synthesis"):
                                st.markdown(analysis)
                                
                        # Show full console trace
                        if captured.getvalue():
                            with st.expander("🔍 Show Detailed Traversal Logs (Cypher queries & LLM outputs)"):
                                st.code(captured.getvalue())
                                
                        # Append assistant message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "analysis": analysis,
                            "actions": actions
                        })
                    except Exception as e:
                        st.error(f"Error during agentic reasoning process: {e}")
                        st.exception(e)
