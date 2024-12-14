import streamlit as st
from st_custom_components import st_audiorec
import openai
from io import BytesIO
import boto3
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

class VoiceAssistant:
    def __init__(self):
        # Initialize OpenAI
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # Initialize AWS Polly
        self.polly = boto3.client(
            'polly',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )

    def process_audio(self, audio_bytes):
        try:
            # Create a BytesIO object
            audio_file = BytesIO(audio_bytes)
            
            # Transcribe using Whisper
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            return transcript["text"]
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None

    def get_ai_response(self, text):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message["content"]
        except Exception as e:
            st.error(f"Error getting AI response: {str(e)}")
            return None

    def text_to_speech(self, text):
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
    st.write("Record a message and I'll respond!")

    assistant = VoiceAssistant()

    # Audio recorder
    audio_bytes = st_audiorec()

    if audio_bytes is not None:
        # Process the audio
        st.info("Processing your message...")
        
        # Get transcription
        transcript = assistant.process_audio(audio_bytes)
        if transcript:
            st.write("You said:", transcript)
            
            # Get AI response
            ai_response = assistant.get_ai_response(transcript)
            if ai_response:
                st.write("Response:", ai_response)
                
                # Convert to speech
                audio_response = assistant.text_to_speech(ai_response)
                if audio_response:
                    st.audio(audio_response, format='audio/mp3')

    # Add a clear button
    if st.button("Clear Conversation"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
