"""
Builds the retrieval reference index: embeds every augmented training clip
with the trained encoder and stores (embedding, label) pairs. At inference
time, a new clip is embedded and matched via cosine-similarity k-NN against
this index instead of being restricted to the fixed softmax classes.
"""
import numpy as np
from keras import models

from config import EMBED_MODEL_PATH, INDEX_PATH, LABELS, X_PATH, Y_PATH


def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    encoder = models.load_model(EMBED_MODEL_PATH)
    embeddings = encoder.predict(X, batch_size=64, verbose=0)

    np.savez(
        INDEX_PATH,
        embeddings=embeddings,
        labels=y,
        label_names=np.array(LABELS),
    )
    print(f"Saved reference index with {len(embeddings)} embeddings -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
