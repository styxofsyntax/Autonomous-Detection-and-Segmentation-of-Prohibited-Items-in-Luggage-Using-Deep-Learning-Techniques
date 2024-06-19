import numpy as np
import tensorflow as tf

X_SIZE, Y_SIZE = (256, 256)
MODEL_PATH = "./models/binary_classification.h5"


def predict(images):
    """
    Perform binary classification on a list of images.

    Args:
        images: Array of images to segment. Input should have shape (None, 256, 256, 3).

    Raises:
        ValueError: If any image in the array does not have the required shape.
    """

    model = tf.keras.models.load_model(MODEL_PATH)

    pred_unsafe = model.predict(images)
    pred_unsafe_threshs = np.where(pred_unsafe > 0.5, 1, 0)

    return pred_unsafe_threshs
