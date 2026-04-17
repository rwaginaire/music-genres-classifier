import streamlit as st
import librosa
import numpy as np
import pickle

# Charger le modèle
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)
    
model = load_model()

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=30)
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
    
    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
    
    # Spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    
    return np.hstack([mfcc, chroma, contrast, tempo])

# Interface
st.title("Music Genre Classifier")
st.write("Upload an audio file to predict its genre.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Sauvegarder temporairement le fichier
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing..."):
        features = extract_features("temp_audio.wav")
        prediction = model.predict([features])[0]

    st.success(f"The predicted genre is: **{prediction}**")