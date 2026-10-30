"""
Raw audio -> fixed-length augmented WAV clips.

Every raw recording (variable length, only a handful of examples per class)
is expanded into many fixed-length (config.DURATION seconds) variants using
noise injection, pitch shift, time-stretch and time-shift, so the downstream
CNN has enough examples per class to train on without instantly overfitting.

Long files (e.g. an ambient background-noise recording) are handled by
taking multiple random crops instead of a single center window, since a
single 4s window would throw away almost all of the recording.
"""
import glob
import os

import librosa
import numpy as np
import soundfile as sf

from config import AUG_DIR, DURATION, LABELS, N_SAMPLES, RAW_DIR, SR

VARIANTS_PER_FILE = 25   # for short clips (eppudra, rajini)
CROPS_PER_LONG_FILE = 20  # for long clips (backgroundnoise)

RNG = np.random.default_rng(42)


def fix_length(y, n_samples=N_SAMPLES):
    """
    Center-crop long clips. Center-pad short clips with a low-level noise
    floor (NOT true digital silence) — a real microphone never records
    true zero amplitude, so padding with exact zeros gives short-utterance
    classes an artificial "silence" signature the CNN can latch onto
    instead of learning the actual sound. That shortcut then fails
    completely on live mic input, which is never truly silent.
    """
    if len(y) < n_samples:
        pad = n_samples - len(y)
        left = pad // 2
        right = pad - left
        floor_std = max(0.02 * np.max(np.abs(y)), 0.0015) if len(y) else 0.0015
        out = RNG.normal(0, floor_std, n_samples)
        out[left:left + len(y)] = y
        return out.astype(np.float32)
    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        return y[start:start + n_samples]
    return y


def random_crop(y, n_samples=N_SAMPLES):
    if len(y) <= n_samples:
        return fix_length(y, n_samples)
    start = RNG.integers(0, len(y) - n_samples)
    return y[start:start + n_samples]


def add_noise(y, snr_db_range=(10, 30)):
    signal_power = np.mean(y ** 2) + 1e-9
    snr_db = RNG.uniform(*snr_db_range)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = RNG.normal(0, np.sqrt(noise_power), y.shape)
    return y + noise


def random_pitch_shift(y, sr, semitone_range=(-2, 2)):
    n_steps = RNG.uniform(*semitone_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def random_time_stretch(y, rate_range=(0.85, 1.15)):
    rate = RNG.uniform(*rate_range)
    return librosa.effects.time_stretch(y, rate=rate)


def random_time_shift(y, max_frac=0.2):
    shift = int(RNG.uniform(-max_frac, max_frac) * len(y))
    return np.roll(y, shift)


def load_background_beds():
    """Real ambient recordings from the backgroundnoise class, used as the
    'room' short utterances get placed into."""
    beds = []
    for f in glob.glob(os.path.join(RAW_DIR, 'backgroundnoise', '*')):
        if f.lower().endswith(('.mp3', '.ogg', '.wav', '.m4a')):
            y, _ = librosa.load(f, sr=SR, mono=True)
            if len(y):
                beds.append(y)
    return beds


def place_in_background(y, beds, rel_range=(0.02, 0.3)):
    """
    Drop the utterance at a random offset inside a DURATION-long window of
    real background noise. Live mic clips are always 4s of real room sound
    with the phrase somewhere in it; without this, short classes (eppudra,
    ~1.4s) were only ever seen padded with synthetic Gaussian hiss, and the
    model learned "clean padding = eppudra, real noise everywhere = rajini"
    — so live input was almost always classified as rajini.
    """
    y = y[:N_SAMPLES]
    bed = random_crop(np.resize(beds[RNG.integers(len(beds))], N_SAMPLES * 2))
    peak = np.max(np.abs(y)) + 1e-9
    bed = bed / (np.max(np.abs(bed)) + 1e-9) * peak * RNG.uniform(*rel_range)
    offset = RNG.integers(0, N_SAMPLES - len(y) + 1)
    bed[offset:offset + len(y)] += y
    return bed.astype(np.float32)


def augment_once(y, sr, beds=None):
    """Apply a random subset of augmentations to one clip."""
    out = y.copy()
    if RNG.random() < 0.7:
        out = random_pitch_shift(out, sr)
    if RNG.random() < 0.7:
        out = random_time_stretch(out)
    if beds and RNG.random() < 0.8:
        out = place_in_background(out, beds)
    elif RNG.random() < 0.5:
        out = random_time_shift(out)
    if RNG.random() < 0.8:
        out = add_noise(out)
    return fix_length(out)


def start_augmentation():
    print(f"Starting augmentation for: {LABELS}")
    beds = load_background_beds()
    print(f"Loaded {len(beds)} background recordings for placement augmentation")

    for label in LABELS:
        src_dir = os.path.join(RAW_DIR, label)
        dst_dir = os.path.join(AUG_DIR, label)
        os.makedirs(dst_dir, exist_ok=True)

        files = [
            f for f in glob.glob(os.path.join(src_dir, '*'))
            if f.lower().endswith(('.mp3', '.ogg', '.wav', '.m4a'))
        ]
        if not files:
            print(f"No raw files found for '{label}' in {src_dir}")
            continue

        n_written = 0
        for filepath in files:
            y, _ = librosa.load(filepath, sr=SR, mono=True)
            duration = len(y) / SR
            stem = os.path.splitext(os.path.basename(filepath))[0]
            stem = "".join(c if c.isalnum() else "_" for c in stem)

            if duration > DURATION * 1.5:
                # long recording: many random crops + light noise, rather
                # than losing most of the audio to a single center window
                for i in range(CROPS_PER_LONG_FILE):
                    clip = random_crop(y)
                    if RNG.random() < 0.6:
                        clip = add_noise(clip, snr_db_range=(15, 35))
                    out_path = os.path.join(dst_dir, f"crop_{i}_{stem}.wav")
                    sf.write(out_path, clip, SR)
                    n_written += 1
            else:
                base = fix_length(y)
                sf.write(os.path.join(dst_dir, f"base_{stem}.wav"), base, SR)
                n_written += 1
                for i in range(VARIANTS_PER_FILE):
                    clip = augment_once(y, SR, beds if label != 'backgroundnoise' else None)
                    out_path = os.path.join(dst_dir, f"aug_{i}_{stem}.wav")
                    sf.write(out_path, clip, SR)
                    n_written += 1

        print(f"{label}: wrote {n_written} clips from {len(files)} raw files")


if __name__ == "__main__":
    start_augmentation()
