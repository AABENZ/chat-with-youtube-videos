import streamlit as st
import os
from dotenv import load_dotenv
from pytube import YouTube
from pydub import AudioSegment
import requests
import json
import openai
import subprocess
import yt_dlp
from datetime import datetime
from pathlib import Path
import assemblyai as aai
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from datetime import datetime, timedelta

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Chat with YouTube Videos",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (moved after set_page_config)
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f0f2f6;
    }
    .question {
        background-color: #e1f5fe;
    }
    .answer {
        background-color: #f5f5f5;
    }
    .video-info {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .loading {
        text-align: center;
        padding: 2rem;
    }
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
session_state_vars = {
    'chat_history': [],
    'transcription_done': False,
    'transcription_file': None,
    'transcript_text': None,
    'processing': False,
    'show_transcription': False,
    'current_question': "",
    'video_info': None
}

for var, default in session_state_vars.items():
    if var not in st.session_state:
        st.session_state[var] = default

# Load environment variables
load_dotenv(".env")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = ASSEMBLYAI_API_KEY

# App header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title('💬 Chat with YouTube Videos')
    st.markdown("---")

st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        Upload a YouTube video link and ask questions about its content!
    </div>
""", unsafe_allow_html=True)

def show_error(message):
    st.error(f"""
        ❌ Error: {message}
        Please try again or contact support if the problem persists.
    """)

def get_video_info(url):
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(url, download=False)
            
            # Convert duration to proper format using timedelta
            duration = video_info.get('duration', 0)
            formatted_duration = str(timedelta(seconds=int(duration))) if duration else "Unknown"
            
            return {
                'title': video_info.get('title', 'Unknown Title'),
                'author': video_info.get('uploader', 'Unknown Author'),
                'length': formatted_duration,
                'thumbnail': video_info.get('thumbnail', None)
            }
    except Exception as e:
        show_error(f"Error fetching video info: {str(e)}")
        return None

# Main functionality
def download_youtube_audio_as_mp3(youtube_url, output_folder="downloads"):
    try:
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"audio_{timestamp}"
        output_path = os.path.join(output_folder, filename)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        return f"{output_path}.mp3"
    except Exception as e:
        show_error(f"Download failed: {str(e)}")
        return None

def transcribe_audio(filename):
    if not os.path.exists(filename):
        show_error(f"File not found: {filename}")
        return None

    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(filename)
        if transcript.status == aai.TranscriptStatus.error:
            show_error(f"Transcription error: {transcript.error}")
            return None
        
        transcription_file = f"{os.path.splitext(filename)[0]}.txt"
        os.makedirs('transcription', exist_ok=True)
        with open(transcription_file, 'w') as f:
            f.write(transcript.text)
        return {"transcription_file": transcription_file, "transcript": transcript.text}
    except Exception as e:
        show_error(f"Transcription failed: {str(e)}")
        return None

def chat_with_llm(transcription_file_path, query):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        loader = TextLoader(transcription_file_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        split_documents = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings
        )
        
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"question": query, "chat_history": []})
        return result["answer"]
    
    except Exception as e:
        show_error(f"Error in chat processing: {str(e)}")
        return None

# Main app layout
with st.container():
    st.subheader("🎥 Video Input")
    user_input = st.text_input("", placeholder="Paste your YouTube URL here...")
    submit_button = st.button("📥 Process Video", use_container_width=True)

if submit_button and user_input:
    st.session_state.processing = True
    
    # Show video info
    video_info = get_video_info(user_input)
    if video_info:
        st.session_state.video_info = video_info
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(video_info['thumbnail'], use_container_width=True)
            with col2:
                st.markdown(f"""
                    ### {video_info['title']}
                    **Channel:** {video_info['author']}  
                    **Duration:** {video_info['length']}
                """)
    
    with st.spinner("🎵 Downloading audio..."):
        filename = download_youtube_audio_as_mp3(user_input)
    
    if filename and os.path.exists(filename):
        with st.spinner("🎯 Transcribing audio..."):
            transcript_result = transcribe_audio(filename)
        
        if transcript_result:
            st.success("✅ Video processed successfully!")
            st.session_state.transcription_done = True
            st.session_state.transcription_file = transcript_result["transcription_file"]
            st.session_state.transcript_text = transcript_result["transcript"]
            st.session_state.show_transcription = True
    
    st.session_state.processing = False

# Show transcription
if st.session_state.show_transcription and st.session_state.transcript_text:
    with st.expander("📝 View Transcription", expanded=True):
        st.write(st.session_state.transcript_text)
        st.download_button(
            label="📥 Download Transcript",
            data=st.session_state.transcript_text,
            file_name="transcript.txt",
            mime="text/plain"
        )

# Chat interface
if st.session_state.transcription_done and not st.session_state.processing:
    st.markdown("### 💭 Chat History")

    # Display chat history
    for idx, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"""
            <div class='chat-message question'>
                <b>Q{idx+1}:</b> {q}
            </div>
            <div class='chat-message answer'>
                <b>A:</b> {a}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🤔 Ask a Question")
    with st.form(key='question_form', clear_on_submit=True):
        user_question = st.text_input("", 
                                    placeholder="What would you like to know about the video?",
                                    key="user_question")
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            submit_question = st.form_submit_button("🚀 Ask", 
                                                  use_container_width=True)
        
        # Move response processing inside the form submission check
        if submit_question and user_question:
            with st.spinner('💭 Thinking...'):
                response = chat_with_llm(st.session_state.transcription_file, user_question)
                if response:
                    st.session_state.chat_history.append((user_question, response))
                    # Display the new question and answer immediately
                    st.markdown(f"""
                        <div class='chat-message question'>
                            <b>Q{len(st.session_state.chat_history)}:</b> {user_question}
                        </div>
                        <div class='chat-message answer'>
                            <b>A:</b> {response}
                        </div>
                    """, unsafe_allow_html=True)
                st.session_state.current_question = ""  # Clear the current question after processing
# Sidebar
with st.sidebar:
    st.markdown("### 📊 Status")
    if st.session_state.transcription_done:
        st.success("✅ Video Processed")
        if st.session_state.transcript_text:
            st.info(f"📝 Transcript Length: {len(st.session_state.transcript_text)} characters")
    
    if st.button("🔄 Reset App", use_container_width=True):
        with st.spinner("Resetting..."):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()