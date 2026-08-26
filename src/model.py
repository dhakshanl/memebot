"""
CNN encoder producing L2-normalized audio embeddings, plus a classifier
head on top. Trained jointly with:
  - softmax cross-entropy on the classifier head (robust supervised signal,
    keeps training stable even with a small dataset)
  - batch-hard triplet loss on the embeddings (pulls same-class embeddings
    together / pushes different-class ones apart), which is what makes the
    embedding space usable for nearest-neighbour *retrieval* against new,
    unseen reference clips later, not just closed-set classification.
"""
import tensorflow as tf
from keras import layers, losses, models

from config import EMBEDDING_DIM, N_FRAMES, N_MELS


def build_encoder(input_shape=(N_MELS, N_FRAMES, 1)):
    inputs = layers.Input(shape=input_shape, name="logmel")

    x = inputs
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(0.15)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(EMBEDDING_DIM)(x)
    embeddings = layers.UnitNormalization(axis=1, name="embedding")(x)

    return models.Model(inputs, embeddings, name="cnn_encoder")


def build_classifier(encoder, num_classes):
    logits = layers.Dense(num_classes, name="logits")(encoder.output)
    return models.Model(encoder.input, [logits, encoder.output], name="cnn_classifier")


def pairwise_distances(embeddings):
    """Euclidean distance matrix between all rows of `embeddings`."""
    dot = tf.matmul(embeddings, embeddings, transpose_b=True)
    sq_norm = tf.linalg.diag_part(dot)
    dist = tf.expand_dims(sq_norm, 1) - 2.0 * dot + tf.expand_dims(sq_norm, 0)
    dist = tf.maximum(dist, 0.0)
    zero_mask = tf.cast(tf.equal(dist, 0.0), tf.float32)
    dist = tf.sqrt(dist + zero_mask * 1e-16) * (1.0 - zero_mask)
    return dist


def batch_hard_triplet_loss(labels, embeddings, margin=0.3):
    """
    'In Defense of the Triplet Loss' (Hermans et al.) batch-hard mining:
    for each anchor in the batch, use the hardest (furthest) same-class
    positive and the hardest (closest) different-class negative.
    """
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
    dist = pairwise_distances(embeddings)

    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    not_self = tf.logical_not(tf.eye(tf.shape(labels)[0], dtype=tf.bool))

    positive_mask = tf.cast(tf.logical_and(labels_equal, not_self), tf.float32)
    hardest_positive = tf.reduce_max(dist * positive_mask, axis=1)

    negative_mask = tf.cast(tf.logical_not(labels_equal), tf.float32)
    max_dist = tf.reduce_max(dist, axis=1, keepdims=True)
    dist_for_negatives = dist + max_dist * (1.0 - negative_mask)
    hardest_negative = tf.reduce_min(dist_for_negatives, axis=1)

    loss = tf.maximum(hardest_positive - hardest_negative + margin, 0.0)
    return tf.reduce_mean(loss)


class BatchHardTripletLoss(losses.Loss):
    def __init__(self, margin=0.3, name="triplet_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        return batch_hard_triplet_loss(y_true, y_pred, margin=self.margin)
