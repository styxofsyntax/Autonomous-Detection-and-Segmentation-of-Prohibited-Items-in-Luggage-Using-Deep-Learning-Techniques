import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# SEMANTIC SEGMENTATION
def apply_color(mask, gun_color = (255, 0, 0), knife_color = (0, 255, 0)):
    colorized = np.zeros((mask.shape[0], mask.shape[1], 3))

    colorized[mask[:, :, 1] == 1] = gun_color
    colorized[mask[:, :, 2] == 1] = knife_color

    return (colorized).astype(np.uint8)


def overlay_mask_on_image(image, mask, alpha=0.4):
    normalized_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) / 255.0

    highlighted_image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

    overlay_image = np.where(normalized_mask[:, :, None] > 0, highlighted_image, image)

    return overlay_image


def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Compute the unweighted loss
        unweighted_loss = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred)

        # Apply the weights to each class
        weight_mask = tf.reduce_sum(weights * y_true, axis=-1)

        # Compute the weighted loss
        weighted_loss = unweighted_loss * weight_mask

        return tf.reduce_mean(weighted_loss)

    return loss


def f1_score(y_true, y_pred):
    # Convert predictions to one-hot encoding
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=y_true.shape[-1])

    # Cast to float32 for calculations
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate true positives, false positives, and false negatives
    tp = K.sum(y_true * y_pred, axis=[0, 1, 2])
    fp = K.sum(y_pred, axis=[0, 1, 2]) - tp
    fn = K.sum(y_true, axis=[0, 1, 2]) - tp

    # Calculate precision and recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    f1 = tf.reduce_mean(f1)  # Average F1 score across classes
    return f1
