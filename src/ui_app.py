"""
Streamlit inspector for the detection pipeline: mic -> log-mel spectrogram
-> CNN embedding -> cosine-similarity comparison against the reference
index. Same recording/scoring logic as live_detect.py, just with every
intermediate stage rendered instead of only the final verdict printed.

Run with: streamlit run ui_app.py   (from the src/ directory)
"""
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import streamlit as st
from keras import models
from sklearn.decomposition import PCA

from config import DURATION, EMBED_MODEL_PATH, N_SAMPLES, SR
from augment import fix_length
from preprocess import clip_to_logmel
from live_detect import (MARGIN_THRESHOLD, SIMILARITY_THRESHOLD,
                          WARMUP_SECONDS, get_mic_id, load_index, retrieve)

st.set_page_config(page_title="Meme Detector Inspector", layout="wide")


@st.cache_resource
def load_assets():
    encoder = models.load_model(EMBED_MODEL_PATH)
    embeddings, labels, label_names = load_index()
    pca = PCA(n_components=2, random_state=0).fit(embeddings)
    return encoder, embeddings, labels, label_names, pca


st.title("Meme Detector Inspector")
st.caption("Mic -> log-mel spectrogram -> CNN embedding -> cosine-similarity comparison")

encoder, ref_embeddings, ref_labels, label_names, pca = load_assets()

devices = sd.query_devices()
input_devices = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
default_idx = get_mic_id()
default_pos = next((pos for pos, (i, _) in enumerate(input_devices) if i == default_idx), 0)

with st.sidebar:
    st.header("Input")
    choice = st.selectbox(
        "Microphone", input_devices, index=default_pos,
        format_func=lambda t: f"{t[0]}: {t[1]}",
    )
    duration = st.slider("Recording length (s)", 1.0, 6.0, float(DURATION), 0.5)
    record_clicked = st.button("Record", type="primary")

if record_clicked:
    mic_id = choice[0]
    with st.spinner(f"Recording {duration:.1f}s from device {mic_id}..."):
        n_total = int((duration + WARMUP_SECONDS) * SR)
        recording = sd.rec(n_total, samplerate=SR, channels=1, device=mic_id)
        sd.wait()
    audio = np.squeeze(recording)[int(WARMUP_SECONDS * SR):]
    audio = fix_length(audio, N_SAMPLES)

    logmel = clip_to_logmel(audio)
    query_embedding = encoder.predict(logmel[np.newaxis, ..., np.newaxis], verbose=0)[0]

    best_label, best_score, second_label, second_score = retrieve(
        query_embedding, ref_embeddings, ref_labels, label_names
    )

    st.session_state["result"] = dict(
        audio=audio, logmel=logmel, embedding=query_embedding,
        best_label=best_label, best_score=best_score,
        second_label=second_label, second_score=second_score,
    )

result = st.session_state.get("result")
if result is None:
    st.info("Pick a microphone and hit Record.")
    st.stop()

audio = result["audio"]
logmel = result["logmel"]
query_embedding = result["embedding"]

peak = float(np.max(np.abs(audio)))
rms = float(np.sqrt(np.mean(audio ** 2)))

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Captured audio")
    st.audio(audio, sample_rate=SR)
    quiet_hint = "  ← very quiet, check mic gain/device" if peak < 0.02 else ""
    st.caption(f"peak={peak:.4f}  rms={rms:.4f}{quiet_hint}")

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(np.linspace(0, len(audio) / SR, len(audio)), audio, linewidth=0.5)
    ax.set_xlabel("seconds")
    ax.set_ylabel("amplitude")
    st.pyplot(fig)

    st.subheader("2. Log-mel spectrogram (model input)")
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(logmel, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax, label="normalized log-power")
    st.pyplot(fig)

with col2:
    st.subheader("3. Embedding vs. reference index (PCA)")
    proj_ref = pca.transform(ref_embeddings)
    proj_query = pca.transform(query_embedding[np.newaxis, :])[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, name in enumerate(label_names):
        mask = ref_labels == idx
        ax.scatter(proj_ref[mask, 0], proj_ref[mask, 1], s=12, alpha=0.5, label=name)
    ax.scatter(*proj_query, marker="*", s=300, c="black", edgecolors="white", label="query", zorder=5)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.subheader("4. Cosine similarity by class")
    sims = ref_embeddings @ query_embedding
    scores = {name: sims[ref_labels == idx].mean() for idx, name in enumerate(label_names)}
    fig, ax = plt.subplots(figsize=(5, 2.5))
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    colors = ["#2ca02c" if n == result["best_label"] else "#888888" for n in names]
    ax.bar(names, vals, color=colors)
    ax.axhline(SIMILARITY_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"threshold {SIMILARITY_THRESHOLD}")
    ax.set_ylabel("mean cosine similarity")
    ax.legend(fontsize=8)
    st.pyplot(fig)

st.subheader("Verdict")
is_confident = (
    result["best_score"] >= SIMILARITY_THRESHOLD
    and (result["best_score"] - result["second_score"]) >= MARGIN_THRESHOLD
)
if result["best_label"] == "backgroundnoise" or not is_confident:
    st.warning(
        f"Background noise / unclear "
        f"(closest: {result['best_label']}={result['best_score']:.3f}, "
        f"runner-up: {result['second_label']}={result['second_score']:.3f})"
    )
else:
    st.success(
        f"DETECTED: {result['best_label'].upper()}  "
        f"({result['best_score'] * 100:.1f}%, runner-up {result['second_label']} "
        f"{result['second_score'] * 100:.1f}%)"
    )
