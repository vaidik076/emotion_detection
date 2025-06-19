import streamlit as st
import numpy as np
import sounddevice as sd
import wavio
import librosa
import os
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder

# Load models
emotion_model = load_model("emotion_detection_upgrademodel.h5")
with open("gender_classifier.pkl", "rb") as f:
    gender_model = pickle.load(f)

# Label encoder
emotion_labels = ['SAD', 'ANG', 'FEA', 'HAP', 'DIS']
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)

# Audio recorder setup
fs = 22050
recording = None

st.set_page_config(page_title="Voice Emotion Detector", layout="centered")
st.title("🎙️ Voice Emotion Detection")

mode = st.radio("Choose Mode", ["Upload Audio File", "Record Voice"])

# Feature extraction
def extract_features(filepath):
    signal, sr = librosa.load(filepath, sr=fs)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Predict emotion
def predict_emotion(filepath):
    features = extract_features(filepath)
    gender_pred = gender_model.predict(features.reshape(1, -1))
    gender = "female" if gender_pred[0] == 1 else "male"

    if gender != "female":
        st.warning("Only female voices are supported for emotion prediction.")
        return

    features_emotion = librosa.feature.mfcc(y=librosa.load(filepath, sr=fs)[0], sr=fs, n_mfcc=40)
    features_emotion = np.mean(features_emotion.T, axis=0)
    emotion_pred = emotion_model.predict(features_emotion.reshape(1, -1))
    emotion_label = label_encoder.inverse_transform([np.argmax(emotion_pred)])
    st.success(f"Predicted Emotion: {emotion_label[0]}")

# Handle upload
if mode == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        with open("uploaded.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.success("File uploaded as uploaded.wav")
        predict_emotion("uploaded.wav")

# Handle record
elif mode == "Record Voice":
    if 'recording' not in st.session_state:
        st.session_state.recording = False

    if not st.session_state.recording:
        if st.button("🟢 Start Recording"):
            st.session_state.recording = True
            st.session_state.audio = sd.rec(int(5 * fs), samplerate=fs, channels=1)
            sd.wait()
            wavio.write("recorded.wav", st.session_state.audio, fs, sampwidth=2)
            st.success("Recording saved as recorded.wav")
            predict_emotion("recorded.wav")
    else:
        if st.button("🔴 Stop Recording"):
            st.session_state.recording = False
