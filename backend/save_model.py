from tensorflow.keras.models import load_model

# Load the existing model
model = load_model('C:/Users/sonuh/visual-damage-detection/backend/defect_detection_model.keras')

# Save the model again in a new format (reshaped)
model.save('C:/Users/sonuh/visual-damage-detection/backend/defect_detection_model_reshaped.keras')

print("Model saved successfully in .keras format")
