import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox
from joblib import load
from features import extract_features
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("ignore", InconsistentVersionWarning)

# Load models
gender_model = load('gender_model.joblib')
emotion_model = load('emotion_model.joblib')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprised', 'sad']

def classify_audio(file_path):
    try:
        features = extract_features(file_path)
        if features is None:
            return "Error extracting features."

        print(f"Features shape: {features.shape}")
        print(f"First few features: {features[:5]}")

        features = np.expand_dims(features, axis=0)

        gender = gender_model.predict(features)[0]
        if gender == 1:
            return "Male audio detected. Please upload a female audio."

        emotion_prediction = emotion_model.predict(features)
        print(f"Emotion model prediction: {emotion_prediction}")

        predicted_emotion = emotion_prediction[0] # The model directly predicts the emotion string

        return f"Detected emotion: {predicted_emotion}"

    except Exception as e:
        return f"Error: {e}"

class AudioClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Audio Emotion Classifier")
        master.configure(bg="#808080")

        self.label = tk.Label(master, text="Select an audio file or record your own:", bg="#808080", font=("Helvetica", 12))
        self.label.pack(pady=10)

        self.select_button = tk.Button(master, text="Select File", command=self.select_file, bg="#36454F", fg="#ffffff", font=("Helvetica", 10))
        self.select_button.pack(pady=10)

        self.record_button = tk.Button(master, text="Record Audio", command=self.record_audio, bg="#36454F", fg="#ffffff", font=("Helvetica", 10))
        self.record_button.pack(pady=10)

        self.detect_button = tk.Button(master, text="Detect Emotion", command=self.detect_emotion, bg="#928E85", fg="#ffffff", font=("Helvetica", 10))
        self.detect_button.pack(pady=10)

        self.result_label = tk.Label(master, text="", bg="#808080", font=("Helvetica", 12))
        self.result_label.pack(pady=10)

        self.selected_file = None

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.selected_file = file_path
            self.result_label.config(text="File selected: " + file_path)
        else:
            messagebox.showerror("Error", "No file selected.")

    def record_audio(self):
        fs = 44100
        seconds = 5
        try:
            self.result_label.config(text="Recording...")
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float64')
            sd.wait()
            temp_file = "recorded_audio.wav"
            sf.write(temp_file, myrecording, fs)
            self.selected_file = temp_file
            self.result_label.config(text="Audio recorded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Recording failed: {e}")

    def detect_emotion(self):
        if self.selected_file:
            result = classify_audio(self.selected_file)
            self.result_label.config(text=result)
        else:
            messagebox.showerror("Error", "No audio file selected or recorded.")

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioClassifierGUI(root)
    root.mainloop()