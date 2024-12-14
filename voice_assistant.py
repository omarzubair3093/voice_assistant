import streamlit as st
import openai
import boto3
from io import BytesIO
import base64

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
            
            // Convert to base64
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = () => {
                const base64data = reader.result;
                audioData.value = base64data;
                
                // Notify Streamlit
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: base64data
                }, "*");
            };
        };

        mediaRecorder.start();
        button.textContent = 'Stop Recording';
        status.textContent = 'Recording...';
        isRecording = true;
    } else {
        // Stop recording
        mediaRecorder.stop();
        button.textContent = 'Start Recording';
        status.textContent = 'Recording stopped';
        isRecording = false;
        
        // Stop all tracks
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
}
</script>

<style>
#recordButton {
    background-color: #FF4B4B;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    margin: 10px 0;
}

#recordButton:hover {
    background-color: #FF3333;
}

#status {
    color: #666;
    margin: 10px 0;
}

#audioPlayback {
    margin: 10px 0;
    width: 100%;
}
</style>
"""

class VoiceAssistant:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.polly = boto3.client(
            'polly',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )

    def process_audio(self, audio_data):
        try:
            # Convert base64 to bytes if needed
            if isinstance(audio_data, str) and audio_data.startswith('data:audio'):
                audio_data = base64.b64decode(audio_data.split(',')[1])
            
            # Create a BytesIO object
            audio_file = BytesIO(audio_data)
            audio_file.name = "recording.wav"  # Whisper needs a filename
            
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
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
    st.write("Record a message or upload an audio file!")

    assistant = VoiceAssistant()

    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

    with tab1:
        st.write("Click the button below to start recording")
        
        # Embed the audio recorder
        audio_recorder = st.components.v1.html(AUDIO_RECORDER_HTML, height=200)
        
        if audio_recorder:  # This will contain the base64 audio data
            if st.button("Process Recording", key="process_recording"):
                with st.spinner("Processing your message..."):
                    # Process the audio
                    transcript = assistant.process_audio(audio_recorder)
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
                                
                                # Download button
                                st.download_button(
                                    label="Download Response",
                                    data=audio_response,
                                    file_name="ai_response.mp3",
                                    mime="audio/mp3"
                                )

    with tab2:
        audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'm4a'])
        if audio_file:
            st.audio(audio_file)
            if st.button("Process Upload", key="process_upload"):
                with st.spinner("Processing your message..."):
                    transcript = assistant.process_audio(audio_file.read())
                    if transcript:
                        st.write("You said:", transcript)
                        response = assistant.get_ai_response(transcript)
                        if response:
                            st.write("Response:", response)
                            audio_response = assistant.text_to_speech(response)
                            if audio_response:
                                st.audio(audio_response, format='audio/mp3')
                                st.download_button(
                                    "Download Response",
                                    audio_response,
                                    "ai_response.mp3",
                                    "audio/mp3"
                                )

    with st.expander("How to use"):
        st.write("""
        **Option 1: Record directly**
        1. Click 'Start Recording'
        2. Allow microphone access if prompted
        3. Speak your message
        4. Click 'Stop Recording'
        5. Click 'Process Recording' to get a response

        **Option 2: Upload audio**
        1. Upload an audio file (WAV, MP3, or M4A format)
        2. Click 'Process Upload' to get a response

        The AI will:
        - Transcribe your audio
        - Generate a response
        - Convert the response to speech
        - Allow you to download the response
        """)

if __name__ == "__main__":
    main()
