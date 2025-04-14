import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Directory paths
female_dir = 'C:/Users/Lutifah/Desktop/Speech-emotion-detection-main/TESS/Female'
male_dir = 'C:/Users/Lutifah/Desktop/Speech-emotion-detection-main/TESS/Male'

# Define emotion labels (update as needed)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprised', 'sad']

# Feature extraction function (returns 32 features)
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)

        # 13 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)

        # 12 Chroma features
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        chroma = np.mean(chroma.T, axis=0)

        # 7 Spectral contrast features
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        contrast = np.mean(contrast.T, axis=0)

        # Total: 13 + 12 + 7 = 32 features
        features = np.concatenate((mfccs, chroma, contrast))
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Process a folder (either male or female) and extract features
def process_folder(folder_path, label):
    features = []
    labels = []

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Extract emotion from the subfolder name
        matched_emotion = None
        for emotion in emotions:
            if emotion in subfolder.lower():
                matched_emotion = emotion
                break

        if not matched_emotion:
            print(f"Skipping folder '{subfolder_path}' (no matching emotion found in name).")
            continue

        emotion_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
        if not emotion_files:
            print(f"No .wav files in {subfolder_path}. Skipping.")
            continue

        print(f"Processing {len(emotion_files)} files in '{subfolder_path}' as '{matched_emotion}'.")

        for file in emotion_files:
            file_path = os.path.join(subfolder_path, file)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(matched_emotion) # Use the extracted emotion label

    return features, labels

# Train gender model
def train_gender_model():
    print("Training gender model...")
    female_features, female_labels = process_folder(female_dir, label=0)  # Female = 0
    male_features, male_labels = process_folder(male_dir, label=1)    # Male = 1

    # Combine features and labels
    features = female_features + male_features
    labels = [0] * len(female_features) + [1] * len(male_features) # Correct gender labels

    if not features:
        print("No features were extracted for the gender model. Check your directory structure and audio files.")
        return

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Gender model classification report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    with open("gender_model.joblib", "wb") as f:
        joblib.dump(model, f)

# Train emotion model
def train_emotion_model():
    print("Training emotion model...")
    female_features, female_labels = process_folder(female_dir, label='female') # Label can be anything here, as the emotion is in the subfolder
    male_features, male_labels = process_folder(male_dir, label='male')   # Label can be anything here

    # Combine features and labels from both genders
    features = female_features + male_features
    labels = female_labels + male_labels

    if not features:
        print("No features were extracted for the emotion model. Check your directory structure and audio files.")
        return

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Emotion model classification report:")
    print(classification_report(y_test, y_pred))

    with open("emotion_model.joblib", "wb") as f:
        joblib.dump(model, f)

# Main function to train both models
def main():
    train_gender_model()
    train_emotion_model()

if __name__ == "__main__":
    main()