import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
from binary_classification import predict as binary_predict
from semantic_segmentation import predict as semantic_predict
from utils import apply_color, overlay_mask_on_image, confusion_matrix_calc, f1_score

IMAGE_PATH = "./data_images/img/"
MASK_PATH = "./data_images/annotations/"
PRED_PATH = "./data_images/pred_masks/"
PLOT_CCOMPARE_PATH = "./data_images/plot_compare"
# IMAGE_PATH = "./test_dataset/img/"
# MASK_PATH = "./test_dataset/annotations/"
# PRED_PATH = "./test_dataset/pred_masks/"

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
    pred_unsafe = binary_predict(images)
    pred_unsafe = np.squeeze(pred_unsafe, axis=-1)

    pred_masks = semantic_predict(images)

    return (pred_unsafe, pred_masks)


def confusion_matrix_disp(label_path, pred_path):
    label_paths = sorted(glob(os.path.join(label_path, "*.png")))
    pred_paths = sorted(glob(os.path.join(pred_path, "*.png")))

    cm = confusion_matrix_calc(label_paths, pred_paths, (576, 768, 3))

    class_labels = ['Background', 'Gun', 'Knife']
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues')
    plt.tight_layout()
    plt.show()


def f1_score_disp(label_path, pred_path):
    label_paths = sorted(glob(os.path.join(label_path, "*.png")))
    pred_paths = sorted(glob(os.path.join(pred_path, "*.png")))

    labels = np.zeros(((len(label_paths), ) + (576, 768, 3)))
    preds = np.zeros(((len(label_paths), ) + (576, 768, 3)))

    for i, (label_path, pred_path) in enumerate(zip(label_paths, pred_paths)):
        label = cv2.imread(label_path)
        pred = cv2.imread(pred_path)

        labels[i] = label
        preds[i] = pred

    return f1_score(labels, preds)


def start():
    images, masks = load_data(IMAGE_PATH, MASK_PATH)
    pred_unsafe, pred_masks = process_images(images)

    # plot image, mask and prediction mask for all predictions

    for i in range(len(images)):

        img = (images[i] * 255).astype(np.uint8)
        orig_mask = apply_color(masks[i], GUN_COLOR, KNIFE_COLOR)

        if pred_unsafe[i] == 1:
            pred_mask = apply_color(pred_masks[i], GUN_COLOR, KNIFE_COLOR)

        else:
            pred_mask = np.zeros_like(orig_mask)

        overlay_orig_mask = overlay_mask_on_image(img, orig_mask, 0.75)
        overlay_pred_mask = overlay_mask_on_image(img, pred_mask, 0.75)

        orig_status = "unsafe"
        pred_status = "unsafe"

        # original mask safe status
        if np.all(np.argmax(masks[i], axis=-1) == 0):
            orig_status = "safe"

        # predcited mask safe status
        if pred_unsafe[i] == 0:
            pred_status = "safe"

        if np.all(np.argmax(pred_masks[i], axis=-1) == 0):
            pred_status = "safe"

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Original Image")
        plt.imshow(img, interpolation="nearest")

        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Original Mask ({})".format(orig_status))
        plt.imshow(overlay_orig_mask, interpolation="nearest")

        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Predicted Mask ({})".format(pred_status))
        plt.imshow(overlay_pred_mask, interpolation="nearest")

        plt.tight_layout()
        plot_path = os.path.join(PLOT_CCOMPARE_PATH, f'compare_{i}.png')
        plt.savefig(plot_path)
        # plt.show()

    # save predictions

    # img_paths = sorted(glob(os.path.join(IMAGE_PATH, "*.png")))

    # for i, (pred_mask, img_path) in enumerate(zip(pred_masks, img_paths)):

    #     if pred_unsafe[i] == 1:
    #         pred_mask = cv2.resize(
    #             pred_mask, (768, 576), interpolation=cv2.INTER_NEAREST)

    #     else:
    #         pred_mask = np.zeros((576, 768, 3))

    #     pred_mask_path = os.path.join(PRED_PATH, os.path.basename(img_path))
    #     pred_mask_colorized = cv2.cvtColor(
    #         apply_color(pred_mask), cv2.COLOR_BGR2RGB)

    #     cv2.imwrite(pred_mask_path, pred_mask_colorized)


start()

# colorized_label_path = "./test_dataset/colorized"
# colorized_pred_path = "./test_dataset/pred_masks"

# confusion_matrix_disp(colorized_label_path, colorized_pred_path)

# with tf.device('/CPU:0'):
#     f1score = f1_score_disp(colorized_label_path, colorized_pred_path)
#     print('\n\n------------------------------------\n')
#     print(f"f1 score: {f1score}")
#     print('\n\n------------------------------------')
