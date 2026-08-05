from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io

app = FastAPI(title="Olive Disease Detection API")

# Charger le modèle une seule fois
model = tf.keras.models.load_model("models/best_model.keras")

class_names = [
    "Healthy",
    "aculus_olearius",
    "olive_peacock_spot"
]

@app.get("/")
def home():
    return {
        "message": "API de détection des maladies des feuilles d'olivier."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)

    classe = np.argmax(prediction)
    confiance = float(np.max(prediction) * 100)

    return {
        "prediction": class_names[classe],
        "confidence": round(confiance, 2)
    }