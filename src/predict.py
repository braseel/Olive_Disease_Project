import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =====================================================
# Charger le modèle
# =====================================================

model = tf.keras.models.load_model("models/best_model.keras")

# Les classes (même ordre que pendant l'entraînement)
class_names = [
    "Healthy",
    "aculus_olearius",
    "olive_peacock_spot"
]

# =====================================================
# Image à tester
# =====================================================

img_path = "dataset/test/image2.jpg"

# =====================================================
# Charger l'image
# =====================================================

img = image.load_img(img_path, target_size=(224, 224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Prétraitement EfficientNet
img_array = preprocess_input(img_array)

# =====================================================
# Prédiction
# =====================================================

prediction = model.predict(img_array, verbose=0)

classe = np.argmax(prediction)
confiance = prediction[0][classe] * 100

print("=" * 60)
print("Image :", img_path)
print("Classe prédite :", class_names[classe])
print(f"Confiance : {confiance:.2f}%")
print("=" * 60)

print("\nProbabilités :")

for i, nom in enumerate(class_names):
    print(f"{nom:20s} : {prediction[0][i]*100:.2f}%")

print("=" * 60)

# =====================================================
# Informations sur le modèle
# =====================================================

params = model.count_params()
size_mb = params * 4 / (1024 ** 2)

print("\nInformations du modèle")
print("----------------------")
print("Nombre de paramètres :", params)
print(f"Taille approximative : {size_mb:.2f} MB")