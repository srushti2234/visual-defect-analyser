from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import cv2
import numpy as np
from database import SessionLocal, DamageAnalysis

app = FastAPI()

# ✅ Basic root endpoint
@app.get("/")
async def root():
    return {"message": "Visual Damage Detection API is running!"}

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.post("/analyze/")
async def analyze_damage(file_name: str, threshold: float):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    damage_score = np.mean(image) / 255 * 100  

    db = SessionLocal()
    analysis = DamageAnalysis(file_name=file_name, damage_percentage=damage_score, threshold=threshold)
    db.add(analysis)
    db.commit()
    db.close()  # Close the database session

    return {
        "file_name": file_name,
        "damage_percentage": damage_score,
        "threshold": threshold,
        "is_defective": damage_score > threshold
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
