import streamlit as st
import os
import re
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Import RAG/LangChain components
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Chatbot using RAG",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    .source-box {
        font-size: 0.85rem;
        background-color: #f1f3f5;
        padding: 0.8rem;
        border-radius: 5px;
        border-left: 3px solid #ff4b4b;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- UTILITY FUNCTIONS -----------------

def extract_video_id(url):
    """
    Extracts the 11-character video ID from various YouTube URL formats.
    """
    if not url:
        return None
    url = url.strip()
    # Check if input is already just an 11-character ID
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
        
    parsed = urlparse(url)
    if parsed.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed.path[1:]
    if parsed.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            p = parse_qs(parsed.query)
            return p.get('v', [None])[0]
        if parsed.path.startswith(('/embed/', '/v/', '/shorts/')):
            parts = parsed.path.split('/')
            if len(parts) >= 3:
                return parts[2]
    # Fallback to general regex
    match = re.search(r'(?:v=|\/shorts\/|\/embed\/|\/v\/|youtu\.be\/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    return None

@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    """
    Loads HuggingFace Embeddings model and caches it across app reruns.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_youtube_video(video_id):
    """
    Fetches transcript, chunks it, and builds a FAISS vector store.
    """
    try:
        # 1. Fetch transcript
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk['text'] for chunk in transcript_list)
        
        # 2. Chunk transcript
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        # 3. Create FAISS Vector Store
        embeddings = load_embeddings_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store, transcript, len(chunks)
        
    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video. Captions must be enabled.")
    except Exception as e:
        raise Exception(f"Error retrieving transcript: {str(e)}")

# ----------------- SIDEBAR -----------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png", width=60)
    st.markdown("### Configuration")
    
    # Groq API Key Input
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        api_status = "Loaded from .env"
        api_input_value = env_key
    else:
        api_status = "Not set"
        api_input_value = ""
        
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=api_input_value,
        placeholder="Enter your gsk_...",
        help=f"Status: {api_status}"
    )
    
    # Model Selection
    model_name = st.selectbox(
        "LLM Model",
        options=["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Select which LLM model to query on Groq"
    )
    
    # Clear Session Button
    if st.button("Reset / Clear Chat", use_container_width=True):
        for key in ["vector_store", "indexed_video_id", "transcript", "messages", "chunk_count"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Provide your **Groq API Key**.
    2. Enter a YouTube URL in the main panel.
    3. Click **Load & Index Video**.
    4. Ask questions in the chat interface!
    """)

# ----------------- MAIN UI -----------------

st.markdown('<div class="main-header">🎥 YouTube Chatbot using RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieve transcript, index with FAISS, and query Groq LLM (LLaMA 3.1) in real-time</div>', unsafe_allow_html=True)

# 1. Video Input Area
st.markdown("### 🔗 1. Enter YouTube Video URL")
col1, col2 = st.columns([3, 1])

with col1:
    url_input = st.text_input(
        "YouTube URL",
        placeholder="e.g. https://www.youtube.com/watch?v=EWvNQjAaOHw",
        label_visibility="collapsed"
    )
    
with col2:
    load_button = st.button("Load & Index Video", type="primary", use_container_width=True)

# Process Video on Button Click or Session State persistence
video_id = extract_video_id(url_input)

if load_button:
    if not groq_api_key:
        st.error("⚠️ Please provide a Groq API Key in the sidebar configuration.")
    elif not video_id:
        st.error("⚠️ Invalid YouTube URL. Please make sure it's a valid link.")
    else:
        # Check if it's already indexed
        if st.session_state.get("indexed_video_id") != video_id:
            with st.status("Processing YouTube video...", expanded=True) as status:
                try:
                    status.update(label="Downloading and parsing YouTube transcript...")
                    vector_store, transcript, chunk_count = process_youtube_video(video_id)
                    
                    status.update(label="Saving embeddings to FAISS vector index...")
                    # Save to session state
                    st.session_state.vector_store = vector_store
                    st.session_state.indexed_video_id = video_id
                    st.session_state.transcript = transcript
                    st.session_state.chunk_count = chunk_count
                    st.session_state.messages = [] # Reset chat history for new video
                    
                    status.update(label="Indexing complete! Ready to answer questions.", state="complete")
                    st.success(f"Successfully processed video. Created {chunk_count} text chunks.")
                except Exception as e:
                    status.update(label="Processing failed!", state="error")
                    st.error(f"Error: {str(e)}")

# 2. Main Workspace (Video Embed and Chatbot)
if "vector_store" in st.session_state:
    st.markdown("---")
    
    workspace_col1, workspace_col2 = st.columns([1, 1])
    
    with workspace_col1:
        st.markdown("### 📺 Video Player")
        st.video(f"https://www.youtube.com/watch?v={st.session_state.indexed_video_id}")
        
        with st.expander("View Full Video Transcript"):
            st.markdown(st.session_state.transcript)
            
    with workspace_col2:
        st.markdown("### 💬 Chat with the Video")
        
        # Initialize message history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Retrieved Sources"):
                        for i, src in enumerate(message["sources"]):
                            st.markdown(f'<div class="source-box"><b>Source {i+1}:</b><br>{src}</div>', unsafe_allow_html=True)
                            
        # Accept user input
        if question := st.chat_input("Ask a question about the video..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
                
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Setup retriever
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    
                    # Setup Prompt
                    prompt = PromptTemplate(
                        template="""
                          You are a helpful assistant.
                          Answer ONLY from the provided transcript context.
                          If the context is insufficient, just say you don't know.

                          {context}
                          Question: {question}
                        """,
                        input_variables=['context', 'question']
                    )
                    
                    # Setup LLM
                    llm = ChatGroq(model=model_name, groq_api_key=groq_api_key)
                    
                    # Fetch retrieved documents for display
                    retrieved_docs = retriever.invoke(question)
                    sources = [doc.page_content for doc in retrieved_docs]
                    
                    # Format context
                    context_text = "\n\n".join(sources)
                    
                    # Run Chain
                    final_prompt = prompt.invoke({"context": context_text, "question": question})
                    
                    with st.spinner("Analyzing transcript..."):
                        response = llm.invoke(final_prompt)
                        answer = response.content
                        
                    message_placeholder.markdown(answer)
                    
                    # Show sources
                    with st.expander("Retrieved Sources"):
                        for i, src in enumerate(sources):
                            st.markdown(f'<div class="source-box"><b>Source {i+1}:</b><br>{src}</div>', unsafe_allow_html=True)
                            
                    # Add to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    message_placeholder.error(f"Error generating answer: {str(e)}")
else:
    # Instructions panel if no video is indexed
    st.markdown("---")
    st.info("ℹ️ Please enter a YouTube URL above and click **Load & Index Video** to start chatting with the video's content.")
