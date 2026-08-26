"""
Augmented WAV clips -> log-mel spectrograms (X.npy / y.npy).

Unlike the old pipeline (which averaged MFCCs over time into a single
40-d vector, discarding all temporal structure), this keeps the full
time-frequency representation as a 2D array so a CNN can actually learn
spatial/temporal patterns instead of being fed a bag-of-coefficients.
"""
import glob
import os

import librosa
import numpy as np

from config import (AUG_DIR, HOP_LENGTH, LABELS, N_FFT, N_FRAMES, N_MELS,
                     SR, X_PATH, Y_PATH)


def clip_to_logmel(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # guard against off-by-one frame counts from edge-case clip lengths
    if log_mel.shape[1] < N_FRAMES:
        pad = N_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant', constant_values=log_mel.min())
    elif log_mel.shape[1] > N_FRAMES:
        log_mel = log_mel[:, :N_FRAMES]

    # per-clip normalization to [-1, 1] so amplitude differences between
    # recordings don't dominate what the CNN learns
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    return log_mel.astype(np.float32)


def extract_features():
    X, y = [], []
    print(f"Feature extraction starting for: {LABELS}")

    for idx, label in enumerate(LABELS):
        label_path = os.path.join(AUG_DIR, label)
        files = glob.glob(os.path.join(label_path, '*.wav'))
        if not files:
            print(f"Warning: no augmented clips found for '{label}' in {label_path}")
            continue

        print(f"Processing {len(files)} files for: {label} (ID: {idx})")
        for filepath in files:
            audio, _ = librosa.load(filepath, sr=SR, mono=True)
            X.append(clip_to_logmel(audio))
            y.append(idx)

    X = np.array(X)[..., np.newaxis]  # add channel dim -> (N, n_mels, frames, 1)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    features, targets = extract_features()
    np.save(X_PATH, features)
    np.save(Y_PATH, targets)
    print(f"Saved X{features.shape} and y{targets.shape}")
