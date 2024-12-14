import streamlit as st
import openai
import boto3
from io import BytesIO
import time

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

    def process_audio(self, audio_file):
        """Process audio using OpenAI Whisper"""
        try:
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
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
    st.write("Upload an audio file or record a message!")

    # Initialize the voice assistant
    assistant = VoiceAssistant()

    # File uploader for audio
    audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'm4a'])

    if audio_file:
        st.audio(audio_file)
        
        if st.button("Process Audio"):
            with st.spinner("Processing your message..."):
                # Reset file pointer
                audio_file.seek(0)
                
                # Transcribe audio
                transcript = assistant.process_audio(audio_file)
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
                            
                            # Option to download response
                            st.download_button(
                                label="Download Response",
                                data=audio_response,
                                file_name="ai_response.mp3",
                                mime="audio/mp3"
                            )

    # Add instructions
    with st.expander("How to use"):
        st.write("""
        1. Upload an audio file containing your message
        2. Click 'Process Audio' to get a response
        3. Listen to the AI's response
        4. Optionally download the response audio
        
        Supported formats: WAV, MP3, M4A
        """)

if __name__ == "__main__":
    main()
