import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("defect_detection_model.keras")

# Define the path to your test image
test_image_path = "C:/Users/sonuh/visual-damage-detection/backend/test_images/img_3.jpg"

# Load and preprocess the image
img = image.load_img(test_image_path, target_size=(224, 224))  # Resize image to match model input
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize image

# Make predictions
predictions = model.predict(img_array)

# Output the predictions
print("Predictions:", predictions)

# If the model is for multi-label classification (e.g., multiple defects), you can interpret the predictions as follows:
# Assuming the model predicts probabilities for each defect type
defect_types = ['Wear and Tear', 'Tear', 'Fraying', 'Dirt']

for i, prediction in enumerate(predictions[0]):
    print(f"{defect_types[i]}: {prediction * 100:.2f}%")
