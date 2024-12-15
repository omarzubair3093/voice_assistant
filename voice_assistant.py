import streamlit as st
import base64
from pydub import AudioSegment
import tempfile

# Error logging for debugging
def log_error(message):
    st.error(message)
    st.write(f"Debug: {message}")

# Page config
st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="wide")

# Custom JavaScript for recording
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
            // Start recording
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayback.src = audioUrl;
                audioPlayback.style.display = 'block';

                // Convert to Base64
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64data = reader.result.split(',')[1];
                    audioData.value = base64data;

                    // Notify Streamlit
                    window.parent.postMessage({ type: "streamlit:setComponentValue", value: base64data }, "*");
                };
            };

            mediaRecorder.start();
            status.textContent = "Recording...";
            button.textContent = "Stop Recording";
            isRecording = true;
        } else {
            // Stop recording
            mediaRecorder.stop();
            status.textContent = "Recording stopped.";
            button.textContent = "Start Recording";
            isRecording = false;
        }
    } catch (error) {
        console.error("Error accessing microphone:", error);
        status.textContent = "Error: Unable to access microphone.";
    }
}
</script>
"""

# Embed JavaScript
st.components.v1.html(AUDIO_RECORDER_HTML, height=300)

# Process recorded audio
recorded_audio_base64 = st.session_state.get("component_value")
if recorded_audio_base64:
    try:
        # Decode Base64 and process audio
        audio_data = base64.b64decode(recorded_audio_base64)
        audio = AudioSegment.from_file(BytesIO(audio_data), format="wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            audio.export(temp_wav_file.name, format="wav")
            st.audio(temp_wav_file.name, format="audio/wav")
            st.success("Recorded audio processed successfully!")
    except Exception as e:
        log_error(f"Error processing recorded audio: {e}")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "m4a"])
if uploaded_file:
    try:
        # Validate and process uploaded audio
        audio = AudioSegment.from_file(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            audio.export(temp_wav_file.name, format="wav")
            st.audio(temp_wav_file.name, format="audio/wav")
            st.success("Uploaded audio processed successfully!")
    except Exception as e:
        log_error(f"Error processing uploaded audio: {e}")
