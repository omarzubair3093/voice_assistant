import streamlit as st
import openai
import boto3
from audiorecorder import audiorecorder
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
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
    </style>
""", unsafe_allow_html=True)

class VoiceAssistant:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Initialize AWS Polly
        self.polly = boto3.client(
            'polly',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )

    def process_audio(self, audio_data):
        """Process audio using OpenAI Whisper"""
        try:
            # Save audio data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                # Transcribe using Whisper
                with open(tmp_file.name, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                return transcript.text
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None

    def get_ai_response(self, text):
        """Get AI response using OpenAI GPT-4"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": text}
                ]
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
            return response['AudioStream'].read()
        except Exception as e:
            st.error(f"Error converting to speech: {str(e)}")
            return None

def main():
    st.title("🎙️ Voice Assistant")
    st.write("Click the microphone to start/stop recording!")

    # Initialize the voice assistant
    assistant = VoiceAssistant()

    # Initialize audio recorder
    audio_recorder = audiorecorder("Click to record", "Recording...")

    # Get the audio data
    audio_data = audio_recorder.get_audio()
    
    if len(audio_data) > 0:
        # Add an audio player to play the recording
        st.audio(audio_data.export().read())
        
        if st.button("Process Recording"):
            with st.spinner("Processing your message..."):
                # Convert audio data to bytes
                audio_bytes = audio_data.export().read()
                
                # Transcribe audio
                transcript = assistant.process_audio(audio_bytes)
                if transcript:
                    st.write("You said:", transcript)
                    
                    # Get AI response
                    response = assistant.get_ai_response(transcript)
                    if response:
                        st.write("Response:", response)
                        
                        # Convert to speech
                        audio_response = assistant.text_to_speech(response)
                        if audio_response:
                            st.audio(audio_response, format='audio/mp3')

if __name__ == "__main__":
    main()
