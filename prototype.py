import streamlit as st # for prototyping
import whisper #gpp speech recognition model
import os
import pyttsx3#for text tospeech conversion easy and oflinre for prototype

# adding FFmpeg path manually because py python couldnt locate it
os.environ["PATH"] += os.pathsep + r"C:/Users/kriti/Videos/ffmpeg-7.1.1-essentials_build/bin" # my path change yours!!!

# loading Whisper model (tiny for faster procrssing(turbo for slower butaccurate) and cpu for processing since no dedicated gpu)
model = whisper.load_model("tiny", device="cpu")

st.title("Audio Transcription Demo with Whisper using Tiny Model")

# file uploader(to upload the audio filre for conversion)
audio_file = st.file_uploader("Upload an Audio file (mp3, wav, m4a, flac) for conversion", type=["mp3", "wav", "m4a","flac"])

if audio_file is not None:
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())

    # preprocessing the audio using only it 30 seconds
    audio = whisper.load_audio("temp_audio")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect language in which audio file is uploades
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    st.write(f"Detected language: {detected_lang}")

    # transcribe the audio into text
    options = whisper.DecodingOptions()
    with st.spinner("Transcribing..."):#(adds a loading text saying Transcribing...)
        result = whisper.decode(model, mel, options)

    #shows result after conversion in .txt format
    st.subheader("Transcription:")
    st.write(result.text)
    
    # now let the user download th extracted textvia download buttion
    st.download_button(
        label="Download description as text",
        data=result.text,
        file_name="Transcribed text",
        mime="text/plain"
    )

st.markdown("---")  # separator line for text to speech and speech to text

st.title("Text to Speech Demo")

# taking text input from user
user_text = st.text_area("Enter text to convert to speech:")

if st.button("Convert to Speech"):
    if user_text.strip() != "":
        engine = pyttsx3.init()# using the pyttsx3 engine for faster conversion and supports offline work
        audio_path = "user_speech.mp3"
        engine.save_to_file(user_text, audio_path)
        engine.runAndWait()
        st.success("Speech saved as user_speech.mp3")

        # play audio in Streamlit for testing
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
    else:
        st.warning("Please enter some text to convert!")#placeholder
