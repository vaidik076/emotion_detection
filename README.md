# emotion_detection
# 🎙️ Emotion Detection from Voice (Female-only) using Deep Learning

This project detects emotions from voice recordings using a deep learning model, with support for both uploading audio files and recording live voice notes through a simple Tkinter GUI. The system also includes a gender classification filter to accept only **female voices** for emotion detection.

---

## 🚀 Features

- 🎧 Detects 5 emotions: `HAPPY`, `SAD`, `ANGRY`, `FEAR`, `DISGUST`
- 🎤 Records live voice (3 seconds)
- 📁 Supports uploading `.wav` voice files
- 👩 Accepts only female voices (male input rejected with a warning)
- 🧠 Trained using the CREMA-D dataset
- 🪄 Built with `TensorFlow`, `Librosa`, `Scikit-learn`, and `Tkinter`

---

## 🗂️ Folder Structure

emotion_detection/
├── AudioWAV/ # Dataset folder (excluded from GitHub)
├── app.py # Main GUI application
├── emotion_detection_model.h5 # Trained emotion model
├── gender_classifier.pkl # Trained gender classification model
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore rules


---

## 🛠️ Installation & Setup

## 1. Clone the repository

git clone https://github.com/vaidik076/emotion_detection.git
cd emotion_detection


2. Create and activate a virtual environment

conda create -n emoenv python=3.9 -y
conda activate emoenv

Install dependencies

pip install -r requirements.txt

▶️ Running the App

python app.py

📦 Model Files
Make sure these are present in the project directory:

emotion_detection_model.h5

gender_classifier.pkl

💡 Note: These are already included in the repo.

📁 Dataset
The dataset (AudioWAV) used during training is not uploaded to GitHub due to size.  
You can download it from [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D) and place it in the `AudioWAV/` folder.


⚠️ Restrictions
 Accepts only female voice inputs.

 Input audio must be in .wav format.

 Recorded audio is limited to 3 seconds by default.

🧪 Model Accuracy
 Emotion detection test accuracy: ~49.49%

 Training accuracy (after 100 epochs): ~59.64%

🧑‍💻 Author
Vaidik Gupta
📫 GitHub: @vaidik076

📜 License
This project is open-source under the MIT License.


