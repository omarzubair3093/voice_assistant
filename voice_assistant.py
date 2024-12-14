import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import openai
import boto3
from datetime import datetime
import queue
import threading
import logging
import av

# Configure page
st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)

class AudioProcessor:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.polly = boto3.client(
            'polly',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )
        self.audio_queue = queue.Queue()
        self.recording = False

    def process_audio(self, audio_data):
        try:
            # Convert to OpenAI format
            response = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data,
                response_format="text"
            )
            return response
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None

    def get_ai_response(self, text):
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
    st.write("An AI voice assistant powered by OpenAI and Amazon Polly")

    processor = AudioProcessor()
    
    # Audio recording interface
    webrtc_ctx = webrtc_streamer(
        key="voice-assistant",
        mode=WebRtcMode.AUDIO_RECEIVER,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = None

    if webrtc_ctx.audio_receiver:
        if not st.session_state["audio_buffer"]:
            st.session_state["audio_buffer"] = []

        def audio_callback(frame: av.AudioFrame):
            sound = frame.to_ndarray()
            st.session_state["audio_buffer"].append(sound)

        webrtc_ctx.audio_receiver.subscribe(audio_callback)

    # Process button
    if st.button("Process Recording"):
        if st.session_state["audio_buffer"]:
            # Convert audio buffer to proper format
            audio_data = np.concatenate(st.session_state["audio_buffer"], axis=0)
            
            # Process the audio
            with st.spinner("Processing audio..."):
                transcript = processor.process_audio(audio_data)
                if transcript:
                    st.success("Transcription complete!")
                    st.write("You said:", transcript)

                    # Get AI response
                    response = processor.get_ai_response(transcript)
                    if response:
                        st.write("AI Response:", response)

                        # Convert to speech
                        audio_response = processor.text_to_speech(response)
                        if audio_response:
                            st.audio(audio_response, format='audio/mp3')

            # Clear the buffer
            st.session_state["audio_buffer"] = []
        else:
            st.warning("No audio recorded yet. Please record something first.")

    # Clear recording button
    if st.button("Clear Recording"):
        st.session_state["audio_buffer"] = []
        st.success("Recording cleared!")

if __name__ == "__main__":
    main()
