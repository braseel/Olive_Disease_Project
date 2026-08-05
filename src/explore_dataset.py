import os
from PIL import Image

# Chemin vers le dossier d'entraînement
train_path = "dataset/train"

print("=" * 50)
print("EXPLORATION DU DATASET")
print("=" * 50)

# Variable pour compter le nombre total d'images
total_images = 0

# Parcourir chaque classe du dossier train
for classe in os.listdir(train_path):

    class_path = os.path.join(train_path, classe)

    # Vérifier que c'est bien un dossier
    if os.path.isdir(class_path):

        print(f"\nClasse : {classe}")

        # Récupérer la liste des images
        images = os.listdir(class_path)

        # Ouvrir la première image de la classe
        image_path = os.path.join(class_path, images[0])
        img = Image.open(image_path)

        # Afficher les informations de l'image
        print(f"Résolution : {img.size}")
        print(f"Mode : {img.mode}")
        print(f"Format : {img.format}")

        # Afficher le nombre d'images
        print(f"Nombre d'images : {len(images)}")

        # Ajouter au total
        total_images += len(images)

# Afficher le nombre total d'images
print("\n" + "=" * 50)
print(f"Nombre total d'images : {total_images}")