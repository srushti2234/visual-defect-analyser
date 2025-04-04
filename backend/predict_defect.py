import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load Trained Model
defect_model = load_model('defect_detection_model.h5')

# Defect Labels
defect_labels = ['Wear and Tear', 'Fraying', 'Tear', 'Corrosion', 'Cracks', 'Dent', 'Bad Welds']

# Adjustable Confidence Threshold
CONFIDENCE_THRESHOLD = 0.3  # Default, can be changed dynamically

def predict_defect(image_path, threshold=CONFIDENCE_THRESHOLD):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict Defects (Multi-Label Classification)
    defect_prediction = defect_model.predict(img_array)
    
    detected_defects = []
    defect_percentages = []

    for i, prob in enumerate(defect_prediction[0]):
        if prob > threshold:
            detected_defects.append(defect_labels[i])
            defect_percentages.append(prob * 100)  # Convert to percentage

    # Display Results
    if detected_defects:
        print("\n🛑 Detected Defects:")
        for defect, percent in zip(detected_defects, defect_percentages):
            print(f"   - {defect}: {percent:.1f}%")

        # Alert if any defect exceeds threshold
        for defect, percent in zip(detected_defects, defect_percentages):
            if percent > (threshold * 100):
                print(f"⚠️ ALERT: {defect} exceeds the set threshold! ({percent:.1f}%)")
    else:
        print("\n✅ No significant defects detected.")

    # Display Image with Labels & Graph
    display_results(image_path, detected_defects, defect_percentages, threshold)

def display_results(image_path, defects, percentages, threshold):
    """Display image with defect details and defect analysis graph side by side."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Create 1 row, 2 columns

    # Show Image with Labels
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    # Add Defect Labels & Percentages on Image
    y_position = 30
    for defect, percent in zip(defects, percentages):
        color = 'red' if percent > (threshold * 100) else 'green'
        axes[0].text(10, y_position, f"{defect}: {percent:.1f}%", color=color, fontsize=12, weight='bold')
        y_position += 25

    # Show Defect Analysis Graph
    if defects:
        bars = axes[1].barh(defects, percentages, color=['red' if p > (threshold * 100) else 'green' for p in percentages])
        axes[1].set_xlabel("Defect Percentage")
        axes[1].set_title("Defect Analysis")
        axes[1].set_xlim(0, 100)

        # Add labels on bars
        for bar, percent in zip(bars, percentages):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{percent:.1f}%", va='center', fontsize=12, weight='bold')

    else:
        axes[1].text(0.5, 0.5, "No defects detected", ha="center", va="center", fontsize=14)
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()

# Test on images
test_images = ['test_images/img_1.jpg', 'test_images/img_2.jpg']
for img_path in test_images:
    print(f"\n==== Analyzing {img_path} ====")
    predict_defect(img_path, threshold=0.4)  # Adjust threshold dynamically
