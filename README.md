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
   - Download the model files separately
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