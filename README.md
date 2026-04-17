# Music Genre Classifier

A machine learning model that predicts the genre of an audio file.

**Live demo** : https://music-genres-classifier.streamlit.app

## How it works
- Audio features extracted with **librosa** (MFCC, Chroma, Spectral Contrast, Tempo)
- Classifier : **SVM** trained on the GTZAN dataset (1000 audio files, 10 genres)
- Accuracy : **68%**

## Tech stack
- Python, scikit-learn, librosa, Streamlit