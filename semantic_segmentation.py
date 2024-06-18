import numpy as np
import tensorflow as tf
from utils import weighted_categorical_crossentropy, f1_score

NUM_CLASSES = 3
X_SIZE, Y_SIZE = (256, 256)
CLASS_WEIGHTS = [0.35595721, 10.59244675, 10.38780955]
MODEL_PATH = "./models/semantic_segmentation.h5"


def predict(images):
    """
    Perform semantic segmentation on a list of images.

    Args:
        images: Array of images to segment. Input should have shape (None, 256, 256, 3).

    Raises:
        ValueError: If any image in the array does not have the required shape.
    """

    if any(img.shape != (X_SIZE, Y_SIZE, 3) for img in images):
        raise ValueError("Image shape should be 256x256x3")

    weights = CLASS_WEIGHTS
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={'loss': weighted_categorical_crossentropy(weights),
                                    'f1_score': f1_score})

    pred_masks = model.predict(images)
    pred_masks_threshs = np.where(pred_masks > 0.5, 1, 0)

    return pred_masks_threshs
