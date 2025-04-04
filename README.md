# Visual Defect Analyzer

A web application for analyzing visual defects in images and videos.

## Project Structure

```
.
├── frontend/           # Frontend web application
│   ├── index.html     # Main HTML file
│   ├── style.css      # CSS styles
│   └── script.js      # JavaScript code
│
└── backend/           # Backend server
    ├── backend.py     # Main backend server
    ├── predict_defect.py      # Defect prediction module
    ├── predict_defect_video.py # Video processing module
    ├── requirements.txt       # Python dependencies
    └── config.py     # Configuration file
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/srushti2234/visual-defect-analyser.git
cd visual-defect-analyser
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
# No additional setup required for the frontend
```

## Training Models from Scratch

The application uses two main models:
1. Object Classification Model
2. Defect Detection Model

### Training the Object Classification Model

1. The dataset is already prepared with the following structure:
   ```
   backend/dataset/
   ├── defective/        # Defective samples
   │   ├── damaged road surface/  # Damaged road images
   │   ├── defective ropes/      # Defective rope images
   │   └── defective wood/       # Defective wood images
   └── non_defective/    # Non-defective samples
       ├── good road surface/    # Good road images
       ├── good ropes/          # Good rope images
       │   ├── undefected cables/  # Good cable images
       │   └── undefected ropes/   # Good rope images
       └── good wood/           # Good wood images
   ```

2. Train the model:
```bash
cd backend
python train_object_model.py
```
   - This will create `object_classification_model.keras` in the backend directory
   - Training progress will be displayed in the console

### Training the Defect Detection Model

1. The dataset is already prepared with the following structure:
   ```
   backend/dataset/
   ├── defective/        # Defective samples
   │   ├── damaged road surface/  # Damaged road images
   │   ├── defective ropes/      # Defective rope images
   │   └── defective wood/       # Defective wood images
   └── non_defective/    # Non-defective samples
       ├── good road surface/    # Good road images
       ├── good ropes/          # Good rope images
       │   ├── undefected cables/  # Good cable images
       │   └── undefected ropes/   # Good rope images
       └── good wood/           # Good wood images
   ```

2. Train the model:
```bash
cd backend
python train_model.py
```
   - This will create `defect_detection_model.h5` in the backend directory
   - Training progress will be displayed in the console

### Model Training Parameters

You can modify the following parameters in the training scripts:

1. In `train_object_model.py`:
   - `IMAGE_SIZE`: Input image dimensions (default: 224x224)
   - `BATCH_SIZE`: Training batch size (default: 32)
   - `EPOCHS`: Number of training epochs (default: 50)
   - `LEARNING_RATE`: Learning rate for the optimizer (default: 0.001)

2. In `train_model.py`:
   - `IMAGE_SIZE`: Input image dimensions (default: 224x224)
   - `BATCH_SIZE`: Training batch size (default: 32)
   - `EPOCHS`: Number of training epochs (default: 50)
   - `LEARNING_RATE`: Learning rate for the optimizer (default: 0.001)

### Notes on Training

1. Training time depends on:
   - Size of your dataset
   - Number of epochs
   - Your hardware (CPU/GPU)
   - Model complexity

2. For best results:
   - Use a balanced dataset
   - Include diverse examples in each class
   - Use data augmentation if you have limited data
   - Consider using GPU acceleration if available

3. Monitor training:
   - Watch the loss and accuracy metrics
   - Use early stopping to prevent overfitting
   - Save model checkpoints periodically

## Running the Application

1. Start the backend server:
```bash
cd backend
python backend.py
```

2. Start the frontend server:
```bash
cd frontend
python -m http.server 8000
```

3. Access the application:
- Open your web browser and go to `http://localhost:8000`

## Important Notes

1. The application requires Python 3.8 or higher
2. Large model files are not included in the repository. You'll need to:
   - Either train the models as described above
   - Or download pre-trained models from a separate source
   - Place them in the `backend` directory
   - The required model files are:
     - `defect_detection_model.h5`
     - `object_classification_model.keras`

## API Endpoints

- `POST /api/analyze-image`: Analyze an image for defects
- `POST /api/analyze-video`: Analyze a video for defects
- `GET /api/results`: Get analysis results

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed correctly
2. Check that the model files are in the correct location
3. Ensure the required directories exist:
   - `backend/uploads/`
   - `backend/graphs/`
4. For training issues:
   - Verify your dataset structure
   - Check available memory/GPU resources
   - Adjust training parameters if needed 