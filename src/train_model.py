import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
# Paramètres
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Charger le dataset d'entraînement
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Charger le dataset de test
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
# Charger EfficientNetB0 pré-entraîné
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Geler les couches
base_model.trainable = False

# Construire le modèle
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax")
])

# Afficher le résumé
model.summary()
# Compiler le modèle
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras",
    save_best_only=True
)
# Entraîner le modèle
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)
# Sauvegarder le modèle final
model.save("models/final_model.keras")

print("Entraînement terminé !")