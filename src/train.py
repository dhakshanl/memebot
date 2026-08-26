import numpy as np
import tensorflow as tf
from keras import losses, optimizers
from sklearn.model_selection import train_test_split

from config import (CLASSIFIER_MODEL_PATH, EMBED_MODEL_PATH, LABELS,
                     MODELS_DIR, X_PATH, Y_PATH)
from model import BatchHardTripletLoss, build_classifier, build_encoder
import os

BATCH_SIZE = 32
EPOCHS = 60


def make_dataset(X, y, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((X, {"logits": y, "embedding": y}))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"Loaded X{X.shape}, y{y.shape}, classes={LABELS}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    encoder = build_encoder(input_shape=X.shape[1:])
    classifier = build_classifier(encoder, num_classes=len(LABELS))
    classifier.summary()

    classifier.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={
            "logits": losses.SparseCategoricalCrossentropy(from_logits=True),
            "embedding": BatchHardTripletLoss(margin=0.3),
        },
        loss_weights={"logits": 1.0, "embedding": 0.5},
        metrics={"logits": "accuracy"},
    )

    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, training=True)
    val_ds = make_dataset(X_val, y_val, BATCH_SIZE, training=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_logits_accuracy", mode="max", patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5,
        ),
    ]

    print("Training the CNN (softmax + batch-hard triplet loss)...")
    classifier.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks,
    )

    val_loss = classifier.evaluate(val_ds, return_dict=True)
    print(f"Validation results: {val_loss}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    encoder.save(EMBED_MODEL_PATH)
    classifier.save(CLASSIFIER_MODEL_PATH)
    print(f"Saved embedding model -> {EMBED_MODEL_PATH}")
    print(f"Saved classifier model -> {CLASSIFIER_MODEL_PATH}")


if __name__ == "__main__":
    main()
