import streamlit as st
from datasets import Dataset, Audio, load_dataset, concatenate_datasets
import pandas as pd
from huggingface_hub import login
import tempfile
from pydub import AudioSegment
import mimetypes #check file types
from dotenv import load_dotenv
import os

# Register missing MIME types (important for .m4a!)
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("audio/mpeg", ".mp3")

load_dotenv()  # Load environment variables from .env file
login(token=os.getenv("IGBO_LLM_KEY"))



def save_temp_file(uploaded_file):
    """Save the uploaded file to a temporary location."""
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(temp_wav.name, "wb") as f:
        f.write(uploaded_file.read())
    audio_path, message = temp_wav.name, "File saved successfully."
    return audio_path, message

# Streamlit app setup
st.set_page_config(page_title="IGBO Dataset Uploader", page_icon=":microphone:", layout="wide")
st.sidebar.title("IGBO Dataset Uploader")
st.sidebar.image("igboimage.jpg", caption="Painting of Igbo People")

st.subheader("Welcome to the IGBO Dataset Uploader!")
st.write("This app allows you to upload audio files and their corresponding text to create a dataset for IGBO TTS (Text-to-Speech) systems.")

st.write("Please ensure your audio files are in .wav format and the text corresponds to the audio content.")

st.write("Enter the corresponding text for the audio file:")
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "gender" not in st.session_state:
    st.session_state.gender = None
if "age" not in st.session_state:
    st.session_state.age= ""
if "dialect" not in st.session_state:
    st.session_state.dialect = ""
if "file_uploader" not in st.session_state:
    st.session_state.file_uploader = None


# Input fields
text_input = st.text_input("Type in the igbo text you want to upload:", key="text_input", placeholder="e.g., Ndewo, Kedu?")
gender = st.selectbox("Select Gender(Optional)", [None,"Female","Male"], key="gender")
age = st.text_input("Enter Age (optional):", placeholder="e.g., 25", key="age")
dialect = st.text_input("Enter Dialect (optional):", placeholder="e.g., anambra", key="dialect")

uploaded_file = st.file_uploader("Choose an audio file (wav, mp3, ogg, flac)", type=["wav", "mp3", "ogg", "flac", "m4a"], key="file_uploader")
if "file_uploader" not in st.session_state:
    st.session_state["file_uploader"] = uploaded_file.read()
st.write("Once you have uploaded the audio file and entered the text, click the 'Upload' button to add them to the dataset.")
final_audio_path = None  # Initialize this variable to avoid errors
#once the file is uploaded
if uploaded_file is not None:
    #check file ext
    st.write(f"Uploaded file: {uploaded_file.name}")
    st.write(f"Detected MIME type: {uploaded_file.type} ")
    # Check if the uploaded file is in .wav format
    check = uploaded_file.type.endswith("wav")
    if not check:
        try:
             # Save uploaded file to temp .m4a
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_m4a:
                tmp_m4a.write(uploaded_file.read())
                tmp_m4a_path = tmp_m4a.name

                st.warning("Converting file to .wav format. Please wait...")

            # Convert to wav
                audio = AudioSegment.from_file(tmp_m4a_path, format="m4a")
                st.success("Conversion to .wav format successful!")
            # Save converted wav to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    audio.export(tmp_wav.name, format="wav")
                    wav_path = tmp_wav.name

            st.success("saved successfully!")
            final_audio_path = wav_path
   
        except Exception as e:
            st.error(f"Error converting file: {e}")
            uploaded_file = None
    else:
        # Already wav → save directly
        final_audio_path, _ = save_temp_file(uploaded_file)
        st.success("Uploaded file is in .wav format.")

    #play + upload
    if final_audio_path:
        st.audio(uploaded_file, format="audio/wav") if uploaded_file is not None else None
        
        if st.button("Upload", on_click=lambda: st.write("Uploading...")):
            st.write("Processing your upload...")
            if  text_input.strip() != "":
               
            
                #load existing dataset if it exists
                try:
                    existing_dataset = load_dataset("kosinebolisa/enudalabs_igbo_tts_dataset")["train"]
                    #existing_df = existing_dataset["train"].to_pandas()
                except Exception as e:
                    st.error(f"Error loading existing dataset: {e}")
                    # If dataset doesn't exist, create an empty DataFrame

                    existing_dataset = pd.DataFrame(columns=["text", "audio_file_path", "gender", "age", "dialect"]) 
                    existing_df = existing_dataset

                # Prepare new row
                new_row = { "text": text_input, "audio_file_path": final_audio_path, "gender": gender if gender else None, "age": age if age else None, "dialect": dialect if dialect else None }
                
                new_dataset = Dataset.from_pandas(pd.DataFrame([new_row]))
                
                # Create and cast new dataset
                #dataset = Dataset.from_pandas(updated_df)
                dataset = new_dataset.cast_column("audio_file_path", Audio(sampling_rate=16000))
                # Merge
                if existing_dataset:
                    updated_dataset = concatenate_datasets([existing_dataset, dataset])
                else:
                     updated_dataset = new_dataset
                updated_dataset.push_to_hub("kosinebolisa/enudalabs_igbo_tts_dataset")

                st.success("File and text uploaded successfully!")
                

            else:
                st.warning("Please enter the corresponding text.")
st.rerun()       