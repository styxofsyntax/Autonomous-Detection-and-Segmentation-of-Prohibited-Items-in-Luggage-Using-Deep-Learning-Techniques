import os
import cv2
import random
import numpy as np
import albumentations as A


def apply_color(mask):
    colorized = np.zeros((mask.shape[0], mask.shape[1], 3))

    colorized[mask == 1] = (0, 0, 255)
    colorized[mask == 2] = (0, 255, 0)

    return (colorized).astype(np.uint8)


def augment(images_to_generate, class_name):
    img_path = f"./test/img/{class_name}"
    # mask_path = f"./test/annotations/{class_name}"

    img_aug_path = f"./aug/img/{class_name}"
    # mask_aug_path = f"./aug/annotations/{class_name}"

    # img_color_path = f"./aug/colorized/{class_name}"

    images = []
    # masks = []

    for img in os.listdir(img_path):
        images.append(os.path.join(img_path, img))

    # for mask in os.listdir(mask_path):
    #    masks.append(os.path.join(mask_path, mask))

    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1)
    ])
    print(f'\nAugmenting... {class_name}')

    count = 0
    for i in range(images_to_generate):

        if i % 50 == 0 and i != 0:
            count += 50
            print(f"done {count}")

        number = random.randint(0, len(images) - 1)
        img = images[number]
       # mask = masks[number]

        original_image = cv2.imread(img)
        original_image = cv2.resize(
            original_image, (768, 576), interpolation=cv2.INTER_NEAREST)
        # original_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

        augmented = aug(image=original_image)
        # augmented = aug(image=original_image, mask=original_mask)
        transformed_image = augmented['image']
        # transformed_mask = augmented['mask']

        new_image_path = os.path.join(
            img_aug_path, "aug_test_img_{}_{}.png".format(class_name, i))
        # new_mask_path = os.path.join(
        #    mask_aug_path, "aug_mask_{}_{}.png".format(class_name, i))
        # new_color_path = os.path.join(
        #   img_color_path, "aug_mask_{}_{}.png".format(class_name, i))

        transformed_image = cv2.resize(
            transformed_image, (768, 576), interpolation=cv2.INTER_NEAREST)
        # transformed_mask = cv2.resize(
        #    transformed_mask, (768, 576), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(new_image_path, transformed_image)
        # cv2.imwrite(new_mask_path, transformed_mask)
        # cv2.imwrite(new_color_path, apply_color(transformed_mask))

    print(f'done {images_to_generate}')


augment(500, 'safe')

# class_names = ["gun", "knife", "safe"]
# for name in class_names:
#     augment(500, name)
