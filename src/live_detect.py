"""
Mic -> log-mel spectrogram -> CNN embedding -> cosine-similarity k-NN
retrieval against the reference index built by build_index.py.

This replaces the old single-vector-softmax-only detector with a proper
embedding retrieval flow: confidence comes from how close the query sits
to real reference clips in embedding space, not just a softmax score.
"""
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import sounddevice as sd
from keras import models

from config import (DURATION, EMBED_MODEL_PATH, INDEX_PATH, LABELS,
                     MEME_SOUNDS_DIR, N_SAMPLES, SR)
from augment import fix_length
from preprocess import clip_to_logmel

TOP_K = 5
SIMILARITY_THRESHOLD = 0.55   # cosine similarity below this -> reject as unknown
MARGIN_THRESHOLD = 0.03       # best label must beat the runner-up by this much

# Some USB audio interfaces emit a burst of noise for the first moment after
# a fresh input stream opens (codec/AGC settling). Audacity hides this because
# you monitor before hitting record; sd.rec() starts capturing immediately,
# so we record a bit extra and drop this prefix before it reaches the model.
WARMUP_SECONDS = 0.4


def get_mic_id(preferred=None):
    """
    Pick an input device. Priority:
    1. `preferred` (index or substring of the device name), if given
    2. whatever Windows currently has set as the default input device
       (this tracks whichever headset/mic is actually active, unlike a
       hardcoded name match)
    3. first device with input channels as a last resort
    """
    devices = sd.query_devices()

    if preferred is not None:
        try:
            idx = int(preferred)
            if devices[idx]['max_input_channels'] > 0:
                return idx
        except ValueError:
            for i, dev in enumerate(devices):
                if preferred.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                    return i
        print(f"Warning: preferred device '{preferred}' not found/not an input device, falling back.")

    default_input = sd.default.device[0]
    if default_input is not None and default_input >= 0 and devices[default_input]['max_input_channels'] > 0:
        return default_input

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            return i
    return None


def get_speaker_id(preferred=None):
    """
    Pick an output device, same priority order as get_mic_id but for
    playback. Windows' "default output device" can silently point at a
    disconnected/unused device (e.g. after switching mics but not
    speakers), so passing `preferred` (matching your actual headset name)
    is the reliable option when meme sounds aren't audible.
    """
    devices = sd.query_devices()

    if preferred is not None:
        try:
            idx = int(preferred)
            if devices[idx]['max_output_channels'] > 0:
                return idx
        except ValueError:
            for i, dev in enumerate(devices):
                if preferred.lower() in dev['name'].lower() and dev['max_output_channels'] > 0:
                    return i
        print(f"Warning: preferred device '{preferred}' not found/not an output device, falling back.")

    default_output = sd.default.device[1]
    if default_output is not None and default_output >= 0 and devices[default_output]['max_output_channels'] > 0:
        return default_output

    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            return i
    return None


def record_clip(duration, mic_id):
    """Record `duration` seconds from `mic_id`, dropping the device warm-up prefix."""
    n_total = int((duration + WARMUP_SECONDS) * SR)
    recording = sd.rec(n_total, samplerate=SR, channels=1, device=mic_id)
    sd.wait()
    audio = np.squeeze(recording)[int(WARMUP_SECONDS * SR):]
    return fix_length(audio, N_SAMPLES)


def play_meme_sound(label, speaker_id=None, wait=True):
    """Play data/meme_sounds/<label>.* if one has been set. Returns whether a
    sound was found; blocks until playback finishes unless wait=False."""
    if not os.path.isdir(MEME_SOUNDS_DIR):
        return False
    matches = glob.glob(os.path.join(MEME_SOUNDS_DIR, f"{label}.*"))
    if not matches:
        return False
    audio, sr = sf.read(matches[0], dtype="float32")
    sd.play(audio, sr, device=speaker_id)
    if wait:
        sd.wait()
    return True


def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Run build_index.py first to create the reference index.")
    data = np.load(INDEX_PATH, allow_pickle=True)
    return data["embeddings"], data["labels"], list(data["label_names"])


def retrieve(query_embedding, embeddings, labels, label_names):
    sims = embeddings @ query_embedding  # cosine similarity (embeddings are L2-normalized)
    top_idx = np.argsort(sims)[::-1][:TOP_K]
    top_sims = sims[top_idx]
    top_labels = labels[top_idx]

    scores = {}
    for name_idx in range(len(label_names)):
        mask = top_labels == name_idx
        scores[label_names[name_idx]] = top_sims[mask].mean() if mask.any() else -1.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    second_label, second_score = ranked[1]
    return best_label, best_score, second_label, second_score


def run_prediction(encoder, embeddings, labels, label_names, mic_id, speaker_id=None):
    duration = DURATION
    print(f"\n--- TAMIL MEME DETECTOR (Mic ID: {mic_id}) ---")
    print(f"Categories: {[l for l in label_names if l != 'backgroundnoise']}")
    input("Press ENTER when you are ready to speak...")

    print(f"RECORDING for {duration}s... (Speak 'eppudra' or 'rajini')")
    n_total = int((duration + WARMUP_SECONDS) * SR)
    recording = sd.rec(n_total, samplerate=SR, channels=1, device=mic_id)
    for i in range(int(duration), 0, -1):
        print(f"Remaining: {i}s", end='\r')
        time.sleep(1)
    sd.wait()
    print("\nRecording captured. Analyzing...")

    audio = np.squeeze(recording)[int(WARMUP_SECONDS * SR):]
    audio = fix_length(audio, N_SAMPLES)
    logmel = clip_to_logmel(audio)[np.newaxis, ..., np.newaxis]
    query_embedding = encoder.predict(logmel, verbose=0)[0]

    best_label, best_score, second_label, second_score = retrieve(
        query_embedding, embeddings, labels, label_names
    )

    print("=" * 40)
    is_confident = best_score >= SIMILARITY_THRESHOLD and (best_score - second_score) >= MARGIN_THRESHOLD
    if best_label == "backgroundnoise" or not is_confident:
        print("RESULT: Just background noise / unclear")
        print(f"(closest: {best_label}={best_score:.3f}, runner-up: {second_label}={second_score:.3f})")
    else:
        print(f"DETECTED: {best_label.upper()}")
        print(f"SIMILARITY: {best_score * 100:.2f}%  (runner-up {second_label}: {second_score * 100:.2f}%)")
        play_meme_sound(best_label, speaker_id)
    print("=" * 40)


def main():
    encoder = models.load_model(EMBED_MODEL_PATH)
    embeddings, labels, label_names = load_index()

    preferred = sys.argv[1] if len(sys.argv) > 1 else None
    mic_id = get_mic_id(preferred)
    speaker_id = get_speaker_id(preferred)
    print(f"Using input device {mic_id}: {sd.query_devices()[mic_id]['name']}")
    print(f"Using output device {speaker_id}: {sd.query_devices()[speaker_id]['name']}")

    while True:
        run_prediction(encoder, embeddings, labels, label_names, mic_id, speaker_id)
        if input("\nTry another one? (y/n): ").lower() != 'y':
            break


if __name__ == "__main__":
    main()
