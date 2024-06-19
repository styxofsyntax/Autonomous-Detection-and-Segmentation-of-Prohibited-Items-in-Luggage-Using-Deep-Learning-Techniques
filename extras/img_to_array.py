import os
import cv2
import numpy as np

data_type = "train"
total_images = 0

dir = f"./{data_type}/img"

SIZE = 256

for root, dirs, files in os.walk(dir):
    total_images += len(files)

classes = os.listdir(dir)
print(classes)
images = np.zeros([total_images, SIZE, SIZE, 3])
img_count = 0
print(images.shape)
labels = np.zeros([total_images, 1])


for i in range(len(classes)):
    image_names = sorted(os.listdir(os.path.join(dir, classes[i])))

    print(len(image_names))
    for j in range(len(image_names)):
        img = cv2.imread(os.path.join(dir, classes[i], image_names[j]))
        img = cv2.resize(img, [SIZE, SIZE])

        img = img/255.0

        labels[img_count] = i
        images[img_count] = img
        img_count += 1


np.save("{}_images_{}.npy".format(data_type, SIZE), images)
np.save("{}_labels_{}.npy".format(data_type, SIZE), labels)

print("LABEL : ", labels[55])
cv2.imshow("image", (images[55] * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
