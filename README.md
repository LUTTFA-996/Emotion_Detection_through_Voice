# Emotion Detection through Voice
This project is a Python-based desktop application that detects emotions from audio recordings using machine learning models. It supports both voice recording and file upload via a user-friendly GUI.

## 🎯 Features
- Detects emotions: Happy, Angry, Sad, Neutral, etc.
- Only accepts **female voices** for emotion classification.
- Alerts if a male voice is uploaded.
- Record your voice or upload a `.wav` file.

  ## 🛠 Tech Stack
- Python 3
- Tkinter (GUI)
- Scikit-learn (ML Models)
- SoundDevice and SoundFile (for recording)
- Librosa (for feature extraction)


## 💾 Models
- `gender_model.joblib`: Binary classifier for gender
- `emotion_model.joblib`: Emotion classifier.

  ## 🚀 Run the App
  python gui.py


  ## 📁 Dataset
Based on Toronto Emotional Speech Set (TESS).


