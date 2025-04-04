import os

DATABASE_URL = "postgresql://postgres:postgre@localhost:5432/damage_db"
UPLOAD_FOLDER = "uploads/"
THRESHOLD = 50  # Default damage threshold

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
