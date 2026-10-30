"""
Soundboard / trigger manager: define a new trigger phrase, record training
takes for it, and set the sound that plays when it's detected live.

Note on the "original/reference" clip: it is deliberately NOT written into
data/raw/<label>/ alongside your own takes. Mixing in audio from a different
source (different mic, room, or a movie clip with background music/reverb)
is exactly what made the `rajini` class hard to detect earlier in this
project (see README) — every augmented training example inherited that
mismatch. The reference clip here is stored separately, purely so you (or
someone else) can listen to what you're supposed to say; it never reaches
the model.

Run via: streamlit run ui_app.py   (this page is auto-discovered from
src/pages/ and shown as a second nav entry)
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import sounddevice as sd
import streamlit as st
from keras import models

from config import (DURATION, EMBED_MODEL_PATH, MEME_SOUNDS_DIR, RAW_DIR,
                     REFERENCE_DIR, SR)
from preprocess import clip_to_logmel
from live_detect import (MARGIN_THRESHOLD, SIMILARITY_THRESHOLD,
                          get_mic_id, get_speaker_id, load_index,
                          record_clip, retrieve)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TAKES = 10

st.set_page_config(page_title="Soundboard", layout="wide")
st.title("Soundboard")
st.caption("Define a trigger phrase, record training takes, and set what plays when it's detected.")


def sanitize_label(name):
    slug = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
    return slug.strip("_")


def current_labels():
    """Fresh scan of data/raw/ — config.LABELS is fixed at import time and
    won't see folders created later in this same long-lived process."""
    if not os.path.isdir(RAW_DIR):
        return []
    return sorted(d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d)))


def find_one(directory, label):
    if not os.path.isdir(directory):
        return None
    matches = glob.glob(os.path.join(directory, f"{label}.*"))
    return matches[0] if matches else None


def play_file(path, speaker_id=None):
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr, device=speaker_id)


# ---------------------------------------------------------------- settings
devices = sd.query_devices()
input_devices = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
output_devices = [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]
default_mic_idx = get_mic_id()
default_mic_pos = next((pos for pos, (i, _) in enumerate(input_devices) if i == default_mic_idx), 0)
default_speaker_idx = get_speaker_id()
default_speaker_pos = next((pos for pos, (i, _) in enumerate(output_devices) if i == default_speaker_idx), 0)

with st.sidebar:
    st.header("Recording settings")
    mic_choice = st.selectbox(
        "Microphone", input_devices, index=default_mic_pos,
        format_func=lambda t: f"{t[0]}: {t[1]}",
    )
    take_seconds = st.slider("Seconds per take", 1.0, 6.0, float(DURATION), 0.5)
    mic_id = mic_choice[0]

    st.header("Playback settings")
    st.caption("Windows' default output can silently point at an unused device after switching mics.")
    speaker_choice = st.selectbox(
        "Speaker / headphone", output_devices, index=default_speaker_pos,
        format_func=lambda t: f"{t[0]}: {t[1]}",
    )
    speaker_id = speaker_choice[0]

# ---------------------------------------------------------- existing triggers
st.header("Existing triggers")
labels = [l for l in current_labels() if l != "backgroundnoise"]

if not labels:
    st.info("No triggers yet — add one below.")
else:
    for label in labels:
        n_takes = len(glob.glob(os.path.join(RAW_DIR, label, "*")))
        ref_path = find_one(REFERENCE_DIR, label)
        meme_path = find_one(MEME_SOUNDS_DIR, label)
        cols = st.columns([2, 2, 2, 2])
        cols[0].markdown(f"**{label}**")
        cols[1].caption(f"{n_takes} training take(s)")
        cols[2].caption("reference set" if ref_path else "no reference clip")
        if meme_path:
            if cols[3].button("Play meme sound", key=f"play_{label}"):
                play_file(meme_path, speaker_id)
        else:
            cols[3].caption("no meme sound set")

st.divider()

# ---------------------------------------------------------- add / update trigger
st.header("Add / update a trigger")
raw_label = st.text_input("Trigger name (e.g. the phrase you'll say)")
label = sanitize_label(raw_label) if raw_label else ""
if raw_label and not label:
    st.error("That name doesn't contain any usable characters.")
elif label:
    st.caption(f"Will be stored as class **{label}**")

    # --- 1. original / reference audio -------------------------------
    st.subheader("1. Original / reference audio")
    st.caption("For your own reference only — never used for training, to avoid baking in mic/room/echo mismatch.")
    ref_tab_upload, ref_tab_record = st.tabs(["Upload", "Record"])

    with ref_tab_upload:
        uploaded_ref = st.file_uploader(
            "Upload the original clip", type=["wav", "mp3", "ogg", "m4a"], key="ref_upload"
        )
        if uploaded_ref is not None:
            os.makedirs(REFERENCE_DIR, exist_ok=True)
            ext = os.path.splitext(uploaded_ref.name)[1] or ".wav"
            ref_path = os.path.join(REFERENCE_DIR, f"{label}{ext}")
            with open(ref_path, "wb") as f:
                f.write(uploaded_ref.getbuffer())
            st.success(f"Saved reference clip -> {ref_path}")

    with ref_tab_record:
        if st.button("Record reference clip now", key="record_ref"):
            with st.spinner(f"Recording {take_seconds:.1f}s..."):
                audio = record_clip(take_seconds, mic_id)
            os.makedirs(REFERENCE_DIR, exist_ok=True)
            ref_path = os.path.join(REFERENCE_DIR, f"{label}.wav")
            sf.write(ref_path, audio, SR)
            st.success(f"Saved reference clip -> {ref_path}")

    existing_ref = find_one(REFERENCE_DIR, label)
    if existing_ref:
        st.audio(existing_ref)

    # --- 2. record training takes -------------------------------------
    st.subheader(f"2. Say it {N_TAKES} times")
    label_dir = os.path.join(RAW_DIR, label)
    existing_takes = sorted(glob.glob(os.path.join(label_dir, "take_*.wav")))
    st.caption(f"{len(existing_takes)}/{N_TAKES} takes recorded so far.")

    if st.button(f"Record {N_TAKES} takes now", type="primary", key="record_takes"):
        os.makedirs(label_dir, exist_ok=True)
        status = st.empty()
        progress = st.progress(0.0)
        for i in range(1, N_TAKES + 1):
            status.info(f"Get ready... take {i}/{N_TAKES}")
            time.sleep(1.2)
            status.warning(f"Recording take {i}/{N_TAKES} — say it now!")
            audio = record_clip(take_seconds, mic_id)
            out_path = os.path.join(label_dir, f"take_{i}.wav")
            sf.write(out_path, audio, SR)
            progress.progress(i / N_TAKES)
            status.success(f"Saved take {i}/{N_TAKES}")
            time.sleep(0.4)
        status.success(f"All {N_TAKES} takes recorded for '{label}'.")
        st.rerun()

    if existing_takes:
        with st.expander(f"Listen to existing takes ({len(existing_takes)})"):
            cols = st.columns(5)
            for i, take_path in enumerate(existing_takes):
                with cols[i % 5]:
                    st.caption(os.path.basename(take_path))
                    st.audio(take_path)

    # --- 3. reward / meme sound -----------------------------------------
    st.subheader("3. Sound to play when detected")
    uploaded_meme = st.file_uploader(
        "Upload the meme/reward sound", type=["wav", "mp3", "ogg", "m4a"], key="meme_upload"
    )
    if uploaded_meme is not None:
        os.makedirs(MEME_SOUNDS_DIR, exist_ok=True)
        ext = os.path.splitext(uploaded_meme.name)[1] or ".wav"
        meme_path = os.path.join(MEME_SOUNDS_DIR, f"{label}{ext}")
        with open(meme_path, "wb") as f:
            f.write(uploaded_meme.getbuffer())
        st.success(f"Saved meme sound -> {meme_path}")

    existing_meme = find_one(MEME_SOUNDS_DIR, label)
    if existing_meme:
        st.audio(existing_meme)

    st.divider()

    # --- build & train ----------------------------------------------------
    st.subheader("Build & train")
    st.caption("Runs augment -> preprocess -> train -> build_index on all current data. Needed before a new/changed trigger is actually detectable.")
    if st.button("Build & train now", type="primary", key="build_train"):
        steps = ["augment.py", "preprocess.py", "train.py", "build_index.py"]
        with st.status("Training pipeline running...", expanded=True) as status_box:
            for step in steps:
                st.write(f"Running {step}...")
                result = subprocess.run(
                    [sys.executable, step], cwd=SRC_DIR, capture_output=True, text=True
                )
                if result.returncode != 0:
                    status_box.update(label=f"{step} failed", state="error")
                    st.code(result.stdout[-3000:] + "\n" + result.stderr[-3000:])
                    st.stop()
            status_box.update(label="Training complete", state="complete")
        st.cache_resource.clear()
        st.success("Model and reference index rebuilt with the latest data.")

st.divider()

# ---------------------------------------------------------------- try it live
st.header("Try it live")


@st.cache_resource
def load_soundboard_assets():
    encoder = models.load_model(EMBED_MODEL_PATH)
    embeddings, ref_labels, label_names = load_index()
    return encoder, embeddings, ref_labels, label_names


try:
    sb_encoder, sb_embeddings, sb_labels, sb_label_names = load_soundboard_assets()
except FileNotFoundError:
    st.info("No trained model yet — record some takes above and hit Build & train.")
else:
    if st.button("Record & test", key="test_record"):
        with st.spinner(f"Recording {take_seconds:.1f}s..."):
            audio = record_clip(take_seconds, mic_id)
        logmel = clip_to_logmel(audio)
        query_embedding = sb_encoder.predict(logmel[np.newaxis, ..., np.newaxis], verbose=0)[0]
        best_label, best_score, second_label, second_score = retrieve(
            query_embedding, sb_embeddings, sb_labels, sb_label_names
        )
        is_confident = (
            best_score >= SIMILARITY_THRESHOLD
            and (best_score - second_score) >= MARGIN_THRESHOLD
        )
        if best_label == "backgroundnoise" or not is_confident:
            st.warning(
                f"Background noise / unclear "
                f"(closest: {best_label}={best_score:.3f}, runner-up: {second_label}={second_score:.3f})"
            )
        else:
            st.success(f"DETECTED: {best_label.upper()}  ({best_score * 100:.1f}%)")
            meme_path = find_one(MEME_SOUNDS_DIR, best_label)
            if meme_path:
                play_file(meme_path, speaker_id)
            else:
                st.caption("(no meme sound set for this trigger yet)")
