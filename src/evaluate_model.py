import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le meilleur modèle
model = tf.keras.models.load_model("models/best_model.keras")

# Charger le dataset de test
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Récupérer les noms des classes
class_names = test_dataset.class_names
print("Classes :", class_names)

# Faire les prédictions
predictions = model.predict(test_dataset)

# Classe prédite
y_pred = np.argmax(predictions, axis=1)

# Vraies classes
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Accuracy
accuracy = np.mean(y_pred == y_true)
print(f"\nAccuracy : {accuracy*100:.2f}%")

# Rapport de classification
print("\nClassification Report :")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion")
plt.show()