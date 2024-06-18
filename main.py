import os
import cv2
import numpy as np
from glob import glob
from utils import apply_color, overlay_mask_on_image
import matplotlib.pyplot as plt
from semantic_segmentation import predict
from tensorflow.keras.utils import to_categorical

IMAGE_PATH = "./data_images/img/"
MASK_PATH = "./data_images/annotations/"
PRED_PATH = "./data_images/pred_masks/"

NUM_CLASSES = 3
X_SIZE, Y_SIZE = (256, 256)
GUN_COLOR = (227, 13, 62)
KNIFE_COLOR = (62, 13, 227)


def load_data(image_path, mask_path):
    """
    Load Images and Masks from provided path.

    Args:
        image_path: path containing images in png format.
        mask_path: path containing masks in png format.

    Returns:
        tuple: (images, masks)
        images shape = (None, X_SIZE, Y_SIZE, 3)
        masks shape = (None, X_SIZE, Y_SIZE, NUM_CLASSES)   
    """

    num_classes = NUM_CLASSES

    img_paths = sorted(glob(os.path.join(image_path, "*.png")))
    print(len(img_paths))
    images = np.zeros([len(img_paths), X_SIZE, Y_SIZE, 3], dtype=np.float32)

    mask_paths = sorted(glob(os.path.join(mask_path, "*.png")))
    masks = np.zeros([len(mask_paths), X_SIZE, Y_SIZE,
                     num_classes], dtype=np.float32)

    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (X_SIZE, Y_SIZE),
                         interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        images[i] = img

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (X_SIZE, Y_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        mask = to_categorical(mask, num_classes=num_classes)
        masks[i] = mask

    return (images, masks)


def process_images(images):
    pred_masks = predict(images)

    return pred_masks


def start():
    images, masks = load_data(IMAGE_PATH, MASK_PATH)
    preds = process_images(images)

    # plot image, mask and prediction mask for all predictions

    for i in range(len(images)):

        img = (images[i] * 255).astype(np.uint8)
        orig_mask = apply_color(masks[i], GUN_COLOR, KNIFE_COLOR)
        pred_mask = apply_color(preds[i], GUN_COLOR, KNIFE_COLOR)
        # orig_mask = apply_color(masks[i])
        # pred_mask = apply_color(preds[i])

        overlay_orig_mask = overlay_mask_on_image(img, orig_mask, 0.75)
        overlay_pred_mask = overlay_mask_on_image(img, pred_mask, 0.75)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Original Image")
        plt.imshow(img, interpolation="nearest")

        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Original Mask")
        plt.imshow(overlay_orig_mask, interpolation="nearest")

        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Predicted Mask")
        plt.imshow(overlay_pred_mask, interpolation="nearest")

        plt.tight_layout()
        plt.show()

    # save predictions

    # img_paths = sorted(glob(os.path.join(IMAGE_PATH, "*.png")))

    # for i, (pred_mask, img_path) in enumerate(zip(preds, img_paths)):
    #     pred_mask = cv2.resize(
    #         pred_mask, (768, 576), interpolation=cv2.INTER_NEAREST)

    #     pred_mask_path = os.path.join(PRED_PATH, os.path.basename(img_path))
    #     pred_mask_colorized = cv2.cvtColor(apply_color(pred_mask), cv2.COLOR_BGR2RGB)

    #     cv2.imwrite(pred_mask_path, pred_mask_colorized)


start()
