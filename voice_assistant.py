import streamlit as st
import openai
import boto3
from io import BytesIO
import base64
from pydub import AudioSegment
import tempfile
import os

# Page config
st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="wide")

# HTML/JavaScript for audio recording
AUDIO_RECORDER_HTML = """
<div>
    <button id="recordButton" onclick="toggleRecording()">Start Recording</button>
    <audio id="audioPlayback" controls style="display: none;"></audio>
    <p id="status"></p>
    <input type="hidden" id="audioData">
</div>

<script>
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

async function toggleRecording() {
    const button = document.getElementById('recordButton');
    const status = document.getElementById('status');
    const audioPlayback = document.getElementById('audioPlayback');
    const audioData = document.getElementById('audioData');

    try {
        if (!isRecording) {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    audioData.value = reader.result.split(',')[1];
                    window.parent.postMessage({ type: "streamlit:setComponentValue", value: reader.result.split(',')[1] }, "*");
                };
            };

            mediaRecorder.start();
            button.textContent = 'Stop Recording';
            status.textContent = 'Recording...';
            isRecording = true;
        } else {
            mediaRecorder.stop();
            button.textContent = 'Start Recording';
            status.textContent = 'Recording stopped.';
            isRecording = false;
        }
    } catch (error) {
        status.textContent = "Error: Unable to access microphone.";
        console.error("Microphone error:", error);
    }
}
</script>
"""

# Main VoiceAssistant class
class VoiceAssistant:
    def __init__(self):
        self.openai_client = openai
        self.polly = boto3.client(
            'polly',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )

    def process_audio(self, audio_data):
        """Process audio by converting and sending to OpenAI Whisper."""
        try:
            audio_bytes = base64.b64decode(audio_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()

                # Use OpenAI's latest transcription endpoint
                with open(temp_file.name, "rb") as audio_file:
                    transcript = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file
                    )

            os.unlink(temp_file.name)
            return transcript["text"]
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None

    def get_ai_response(self, text):
        """Get AI response using OpenAI GPT."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": text}]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error getting AI response: {e}")
            return None

    def text_to_speech(self, text):
        """Convert text to speech using AWS Polly."""
        try:
            response = self.polly.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId="Joanna",
                Engine="neural"
            )
            return response["AudioStream"].read()
        except Exception as e:
            st.error(f"Error converting to speech: {e}")
            return None

def main():
    st.title("🎙️ Voice Assistant")
    assistant = VoiceAssistant()

    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    # Tab 1: Record Audio
    with tab1:
        st.components.v1.html(AUDIO_RECORDER_HTML, height=300)
        audio_data = st.session_state.get("component_value")
        if audio_data and st.button("Process Recording"):
            with st.spinner("Processing..."):
                transcript = assistant.process_audio(audio_data)
                if transcript:
                    st.write("You said:", transcript)
                    response = assistant.get_ai_response(transcript)
                    if response:
                        st.write("Response:", response)
                        tts_audio = assistant.text_to_speech(response)
                        if tts_audio:
                            st.audio(tts_audio, format="audio/mp3")

    # Tab 2: Upload Audio
    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
        if audio_file and st.button("Process Upload"):
            with st.spinner("Processing..."):
                try:
                    # Use OpenAI Whisper transcription for file upload
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                        temp_audio.write(audio_file.read())
                        temp_audio.flush()
                        with open(temp_audio.name, "rb") as audio:
                            transcript = openai.Audio.transcribe(
                                model="whisper-1",
                                file=audio
                            )
                        st.write("You said:", transcript['text'])
                        response = assistant.get_ai_response(transcript['text'])
                        if response:
                            st.write("Response:", response)
                            tts_audio = assistant.text_to_speech(response)
                            if tts_audio:
                                st.audio(tts_audio, format="audio/mp3")
                except Exception as e:
                    st.error(f"Error processing upload: {e}")
                finally:
                    os.unlink(temp_audio.name)

if __name__ == "__main__":
    main()
