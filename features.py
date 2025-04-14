import librosa
import numpy as np

# Extract 32 features: 13 MFCC + 12 Chroma + 7 Spectral Contrast
def extract_features(file_path):
    y, sr = librosa.load(file_path)

    # 13 MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

    # 12 Chroma
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    # 7 Spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)

    # Combined features (13 + 12 + 7 = 32)
    return np.concatenate((mfccs, chroma, contrast))
