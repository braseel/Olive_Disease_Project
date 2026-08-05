import os
import random
import matplotlib.pyplot as plt
from PIL import Image

train_path = "dataset/train"

classes = os.listdir(train_path)

plt.figure(figsize=(20, 10))

image_number = 1

for classe in classes:

    class_path = os.path.join(train_path, classe)

    images = os.listdir(class_path)

    random_images = random.sample(images, 10)

    for image_name in random_images:

        image_path = os.path.join(class_path, image_name)

        img = Image.open(image_path)

        plt.subplot(3, 10, image_number)
        plt.imshow(img)
        plt.title(classe)
        plt.axis("off")

        image_number += 1

plt.tight_layout()
plt.show()