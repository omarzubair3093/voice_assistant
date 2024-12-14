import streamlit as st
import numpy as np
import openai
import boto3
from streamlit_webrtc import webrtc_streamer
import av
import queue
import threading
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide"
)

class AudioProcessor:
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
        
        # Initialize audio buffer
        self.audio_frames = []
        self.audio_buffer = queue.Queue()

    def audio_frame_callback(self, frame):
        """Process incoming audio frames"""
        try:
            audio_data = frame.to_ndarray()
            self.audio_frames.append(audio_data)
        except Exception as e:
            st.error(f"Error processing audio frame: {str(e)}")

    def process_audio(self, audio_data):
        """Convert audio to text using OpenAI Whisper"""
        try:
            response = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data,
                response_format="text"
            )
            return response
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
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
    st.write("Speak into your microphone and I'll respond!")

    # Initialize processor
    processor = AudioProcessor()

    # WebRTC Audio configuration
    webrtc_ctx = webrtc_streamer(
        key="voice-assistant",
        audio_receiver_size=1024,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": False,
            "audio": True
        }
    )

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = []

    # Handle audio input
    if webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames()
            for audio_frame in audio_frames:
                sound = audio_frame.to_ndarray()
                st.session_state["audio_buffer"].append(sound)
        except queue.Empty:
            pass

    # Process recording button
    if st.button("Process Recording"):
        if st.session_state["audio_buffer"]:
            with st.spinner("Processing your message..."):
                # Combine audio frames
                audio_data = np.concatenate(st.session_state["audio_buffer"])
                
                # Transcribe audio
                transcript = processor.process_audio(audio_data)
                if transcript:
                    st.write("You said:", transcript)
                    
                    # Get AI response
                    response = processor.get_ai_response(transcript)
                    if response:
                        st.write("Response:", response)
                        
                        # Convert to speech
                        audio_response = processor.text_to_speech(response)
                        if audio_response:
                            st.audio(audio_response, format='audio/mp3')
            
            # Clear the buffer
            st.session_state["audio_buffer"] = []
        else:
            st.warning("No audio recorded. Please speak into your microphone first.")

    # Clear button
    if st.button("Clear Recording"):
        st.session_state["audio_buffer"] = []
        st.success("Recording cleared!")

if __name__ == "__main__":
    main()
