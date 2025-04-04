from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = FastAPI()

# ✅ Fix CORS issue by allowing all origins (Frontend can communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Change "*" to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model
try:
    object_model = load_model("object_classification_model.keras")
    defect_model = load_model("defect_detection_model.h5")
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")

# ✅ Labels
object_labels = ["Road Surface", "Iron", "Rope", "Wood"]
defect_labels = ["Wear and Tear", "Fraying", "Tear", "Corrosion", "Cracks", "Dent", "Bad Welds"]

# ✅ Confidence Threshold
CONFIDENCE_THRESHOLD = 0.3

def process_image(image_bytes):
    """Preprocess image for model prediction."""
    img = image.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def root():
    return {"message": "Visual Damage Detection API is running!"}

@app.post("/predict-defect/")
async def predict_defect(file: UploadFile = File(...)):
    """Receives an image, processes it, and returns detected defects as JSON."""
    try:
        image_bytes = await file.read()
        img_array = process_image(image_bytes)

        # ✅ Predict object type
        object_prediction = object_model.predict(img_array)
        object_class = np.argmax(object_prediction)
        object_name = object_labels[object_class]

        # ✅ Predict defects
        defect_prediction = defect_model.predict(img_array)[0]

        detected_defects = [
            {"defect": defect_labels[i], "confidence": round(float(prob * 100), 2)}
            for i, prob in enumerate(defect_prediction) if prob > CONFIDENCE_THRESHOLD
        ]

        return {
            "filename": file.filename,
            "object_detected": object_name,
            "detected_defects": detected_defects,
            "message": "Defects detected" if detected_defects else "No defects detected"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
