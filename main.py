from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# CORS (Frontend ulanadigan bo‘lsa)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Modelni yuklash
MODEL_PATH = "malaria_final_best_v11.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 128

# ✅ Preprocessing funksiyasi
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("❌ Rasm faylini ochib bo‘lmadi. Iltimos, haqiqiy rasm yuboring.")
        
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Predict API
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img_array = preprocess_image(contents)

        prediction = model.predict(img_array)[0][0]
        label = "Parasitized" if prediction > 0.5 else "Uninfected"
        confidence = round(float(prediction if label == "Parasitized" else 1 - prediction), 4)

        return JSONResponse(content={
            "prediction": label,
            "confidence": confidence
        })

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Serverda xatolik: {str(e)}"})

