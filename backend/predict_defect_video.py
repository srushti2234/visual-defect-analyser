import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load Trained Models
object_model = load_model('object_classification_model.keras')  # Object Classification Model
defect_model = load_model('defect_detection_model.h5')  # Defect Detection Model

# Labels
object_labels = ['Road Surface', 'Iron', 'Rope', 'Wood']  # Adjust based on dataset
defect_labels = ['Wear and Tear', 'Fraying', 'Tear', 'Corrosion', 'Cracks', 'Dent', 'Bad Welds']

# Adjustable Confidence Threshold
CONFIDENCE_THRESHOLD = 0.3  # Default threshold, can be changed dynamically

def predict_defect_video(video_path, threshold=CONFIDENCE_THRESHOLD):
    cap = cv2.VideoCapture(video_path)
    
    frame_captures = []
    all_defects = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        # Resize and preprocess frame
        frame_resized = cv2.resize(frame, (224, 224))
        frame_array = frame_resized / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        # Predict Object Type
        object_prediction = object_model.predict(frame_array)
        object_class = np.argmax(object_prediction)
        object_name = object_labels[object_class]  # Get Object Name

        # Predict Defects
        defect_prediction = defect_model.predict(frame_array)

        detected_defects = []
        defect_percentages = []

        for i, prob in enumerate(defect_prediction[0]):
            if prob > threshold:
                detected_defects.append(defect_labels[i])
                defect_percentages.append(prob * 100)  # Convert to percentage
                
                # Store defect percentage for final graph
                if defect_labels[i] in all_defects:
                    all_defects[defect_labels[i]].append(prob * 100)
                else:
                    all_defects[defect_labels[i]] = [prob * 100]

        # Annotate Frame
        if detected_defects:
            annotated_frame = frame.copy()
            y_offset = 40
            cv2.putText(annotated_frame, f"Object: {object_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for i, (defect, percent) in enumerate(zip(detected_defects, defect_percentages)):
                color = (0, 0, 255) if percent > (threshold * 100) else (255, 255, 0)  # Red for alert
                cv2.putText(annotated_frame, f"{defect}: {percent:.1f}%", (10, 60 + (i * 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Alert if any defect exceeds the threshold
                if percent > (threshold * 100):
                    print(f"⚠️ ALERT: {defect} exceeds threshold! ({percent:.1f}%)")

            frame_captures.append(annotated_frame)

    cap.release()
    cv2.destroyAllWindows()

    # Display all processed frames with labels
    display_results(frame_captures, all_defects, threshold)

def display_results(frames, defect_data, threshold):
    """Displays final analysis with all captured frames & defect graph."""
    num_frames = len(frames)

    if num_frames == 0:
        print("No significant defects detected in the video.")
        return

    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 3, 8))

    fig.suptitle("Defect Detection Analysis", fontsize=16, fontweight="bold")

    # Display each captured frame
    for i in range(num_frames):
        frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(frame_rgb)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Frame {i+1}")

    # Create defect analysis graph
    defect_types = list(defect_data.keys())
    avg_percentages = [np.mean(vals) for vals in defect_data.values()]

    bars = axes[1, 0].barh(defect_types, avg_percentages, color=['red' if p > (threshold * 100) else 'blue' for p in avg_percentages])
    axes[1, 0].set_xlabel("Defect Percentage")
    axes[1, 0].set_title("Defect Analysis")
    axes[1, 0].set_xlim(0, 100)

    for bar, percent in zip(bars, avg_percentages):
        axes[1, 0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{percent:.1f}%", 
                      va='center', fontsize=12, weight='bold')

    # Hide extra axes
    for j in range(1, num_frames):
        axes[1, j].axis("off")

    plt.tight_layout()
    plt.show()

# Test on video
video_path = 'test_videos/sample_video.mp4'
predict_defect_video(video_path, threshold=0.4)
