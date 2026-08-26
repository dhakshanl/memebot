# memebot

Voice-triggered meme sound detector. Speak a trigger phrase into a
microphone and it's recognized via a CNN embedding + nearest-neighbour
retrieval, instead of a fixed closed-set classifier — so new reference
clips can be added later without retraining from scratch.

## How it works

```
data/raw/<label>/*        raw recordings, a handful per class
      -> augment.py       pitch/tempo/noise/time-shift augmentation
      -> data/augmented/  many fixed-length WAV variants per class
      -> preprocess.py    log-mel spectrograms -> X.npy / y.npy
      -> train.py         CNN encoder + classifier head
      -> build_index.py   embeds every augmented clip -> reference index
      -> live_detect.py   mic -> embedding -> cosine-similarity k-NN
```

The encoder ([src/model.py](src/model.py)) is trained with two losses at once:
softmax cross-entropy on a classifier head, and batch-hard triplet loss
on the embeddings themselves. The triplet loss is what makes the
embedding space usable for retrieval against reference clips at
inference time, rather than being limited to the classes seen during
training.

Classes (see [src/config.py](src/config.py)): `backgroundnoise`, `eppudra`, `rajini`.

## Setup

```
pip install tensorflow keras librosa soundfile sounddevice scikit-learn numpy matplotlib streamlit
```

## Usage

Run from the `src/` directory. Only needed end-to-end if you change
`data/raw/` — trained models and the reference index are already
committed in `models/`.

```
cd src
python augment.py        # data/raw -> data/augmented
python preprocess.py     # data/augmented -> X.npy / y.npy
python train.py          # trains and saves models/*.keras
python build_index.py    # builds models/reference_index.npz
```

**Live detection** (mic -> prediction):

```
python live_detect.py [device_name_or_index]
```

Records a fixed-length clip, embeds it, and matches it against the
reference index. Rejects as "background noise / unclear" unless the
best match clears both a similarity threshold and a margin over the
runner-up (see `SIMILARITY_THRESHOLD` / `MARGIN_THRESHOLD` in
[src/live_detect.py](src/live_detect.py)).

**Inspector UI** (see every pipeline stage — waveform, spectrogram,
embedding vs. reference index, per-class similarity):

```
streamlit run ui_app.py
```

## Notes

- Detection quality is sensitive to the recording device — a mismatch
  between the mic used for `data/raw/` and the one used live can hurt
  accuracy (different USB audio interfaces can introduce their own
  noise/frequency-response artifacts the model never saw in training).
- Classes built from very few raw takes (especially longer, variable-
  length phrases) generalize less reliably than short, consistently-
  timed words — more/varied raw recordings per class is the most
  effective fix if a class underperforms.
