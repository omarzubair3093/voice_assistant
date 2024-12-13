import streamlit as st
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
import openai
import boto3
import os
from datetime import datetime
import base64

# Configure page
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for dark theme
st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #2E2E2E;
            color: #FFFFFF;
        }
        .stSlider {
            background-color: #2E2E2E;
        }
    </style>
""", unsafe_allow_html=True)

class CloudVoiceAssistant:
    def __init__(self):
        # Initialize OpenAI client
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # Initialize AWS client
        self.polly = boto3.client(
            'polly',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )

    def transcribe_audio(self, audio_bytes):
        """Transcribe audio using OpenAI Whisper"""
        try:
            # Save audio bytes to a temporary file
            temp_audio = BytesIO(audio_bytes)
            temp_audio.name = "temp.wav"
            
            # Transcribe using OpenAI Whisper
            transcript = openai.Audio.transcribe(
                "whisper-1",
                temp_audio
            )
            return transcript.text
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None

    def get_ai_response(self, text):
        """Get AI response using OpenAI GPT"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error getting AI response: {str(e)}")
            return None

    def text_to_speech(self, text):
        """Convert text to speech using Amazon Polly"""
        try:
            response = self.polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId='Joanna',
                Engine='neural'
            )
            
            # Convert the audio stream to bytes
            audio_data = response['AudioStream'].read()
            return audio_data
        except Exception as e:
            st.error(f"Error converting text to speech: {str(e)}")
            return None

def main():
    st.title("🎙️ Voice Assistant")
    st.write("An interactive voice assistant powered by OpenAI and Amazon Polly")

    # Initialize assistant
    assistant = CloudVoiceAssistant()

    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Voice Settings")
    st.sidebar.write("Using Amazon Polly Neural Engine")

    # Main interface
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Record New Message")
        # Using audio_recorder component
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e01b24",
            neutral_color="#ffffff"
        )

        if audio_bytes:
            st.session_state['audio_bytes'] = audio_bytes
            st.success("Recording saved! Processing...")
            
            # Transcribe audio
            transcript = assistant.transcribe_audio(audio_bytes)
            if transcript:
                st.session_state['transcript'] = transcript
                st.write("Transcript:", transcript)

                # Get AI response
                ai_response = assistant.get_ai_response(transcript)
                if ai_response:
                    st.session_state['ai_response'] = ai_response
                    
                    # Convert AI response to speech
                    audio_response = assistant.text_to_speech(ai_response)
                    if audio_response:
                        st.session_state['audio_response'] = audio_response

    with col2:
        st.subheader("Assistant Response")
        if 'ai_response' in st.session_state:
            st.write("AI Response:", st.session_state['ai_response'])
            
        if 'audio_response' in st.session_state:
            st.audio(st.session_state['audio_response'], format='audio/mp3')

    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        for key in ['audio_bytes', 'transcript', 'ai_response', 'audio_response']:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
