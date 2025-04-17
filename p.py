import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
import whisper
import sounddevice as sd
import numpy as np
import pyttsx3
import scipy.io.wavfile as wavfile
import tempfile
import soundfile as sf
import os
import asyncio
# Ensure a running event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Suppress Whisper warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Disable Streamlit file watcher
os.environ["STREAMLIT_WATCH_FILE"] = "false"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"


st.set_page_config(page_title="Voice Chatbot", layout="wide")

# Sidebar configuration
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox("Choose a Model:", ["deepseek-r1:1.5b"])

# Load CSS
st.markdown("""<style>
.chat-container {
    background-color: #e3f2fd;
    padding: 20px;
    border-radius: 10px;
}
.user-message {
    background-color: #bbdefb;
    color: black;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 5px;
}
.bot-message {
    background-color: #1e88e5;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 5px;
}
</style>""", unsafe_allow_html=True)

st.title("🎤 Voice Chatbot")

# Load whisper model once
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Setup Ollama LLM
llm_engine = ChatOllama(
   model=model_choice,
   base_url="http://localhost:11434",
   temperature=0.3
)

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI expert coding assistant. Provide concise, correct solutions with strategic print statements for debugging. Always respond in English."
)

# Text-to-speech
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()

# Voice recording + Whisper transcription
def record_and_transcribe():
    try:
        duration = 5  # seconds
        sample_rate = 16000
        local_audio_path = os.path.join(os.getcwd(), "recorded.wav")

        st.info("Recording for 5 seconds...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        st.success("Recording complete!")

        # Save recording locally
        sf.write(local_audio_path, recording, sample_rate)
        st.info(f"Audio saved as {local_audio_path}")
        print(f"Audio saved as {local_audio_path}")
        # Transcribe
        result = whisper_model.transcribe(r"C:\\Users\\Prsd0\\interactiveChatbot\\recorded.wav")
        st.success("Transcription complete!")
        return result["text"]

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Session state
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm deepseek"}]

chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input method selector
input_method = st.radio("Input method", ["Text", "Voice"], horizontal=True)
user_query = None

if input_method == "Text":
    user_query = st.chat_input("Type your coding question here...")
elif input_method == "Voice":
    if st.button("🎤 Speak"):
        user_query = record_and_transcribe()
        st.success(f"You said: {user_query}")

# Prompt building
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Generate AI response
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Handle conversation
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("🤖 Thinking..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    speak(ai_response)

# Refresh chat
st.rerun()
