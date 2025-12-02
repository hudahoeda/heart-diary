# Heart Diary 💓

A FastAPI web application for analyzing Polar H10 ECG data and tracking heart health over time.

## Features

- **Upload ECG Data**: Upload Polar H10 ECG logs, accelerometer data, and symptom markers
- **Automated Analysis**: Generates clinical-style ECG reports with:
  - Heart rate statistics (mean, max, min, HRV)
  - Ectopic burden calculation
  - Rhythm event detection (couplets, bigeminy, trigeminy)
  - Baseline morphology (QRS, QTc)
  - Motion/activity detection from accelerometer data
  - Symptom marker visualization
- **Calendar View**: Browse historical ECG recordings by date
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the FastAPI server:

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

Then open your browser to: **http://localhost:8000**

## Usage

### Uploading ECG Data

1. Go to the home page (`/`)
2. Select your ECG file (required) - Polar H10 export `.txt` or `.csv`
3. Optionally add:
   - Accelerometer log for motion/activity detection
   - Marker log for symptom annotations
4. Add any notes about the recording session
5. Click "Analyze ECG Data"

### Viewing Reports

- Access the **Calendar** (`/calendar`) to see all recordings organized by date
- Click on any report badge in the calendar to view the full analysis
- Use the report list below the calendar for a chronological view

### File Formats

The app expects Polar H10 Sensor Logger export files:

- **ECG**: `Polar_H10_XXXXXX_YYYYMMDD_HHMMSS_ECG.txt`
- **Accelerometer**: `Polar_H10_XXXXXX_YYYYMMDD_HHMMSS_ACC.txt`
- **Markers**: `MARKER_YYYYMMDD_HHMMSS.txt`

## Project Structure

```
heart-diary/
├── main.py              # FastAPI application
├── ecg_processor.py     # ECG analysis module (from polar.py)
├── requirements.txt     # Python dependencies
├── templates/           # HTML templates
│   ├── index.html       # Upload page
│   └── calendar.html    # Calendar view
├── static/              # Static files (CSS, JS)
├── uploads/             # Uploaded files (temporary)
├── reports/             # Generated HTML reports
└── data/
    └── reports.json     # Report metadata database
```

## Medical Disclaimer

⚠️ This application is for **informational purposes only** and is NOT a medical device. 

- Reports identify intervals consistent with premature beats but cannot definitively diagnose conditions
- QRS/QT values are estimates and may not be accurate
- Always consult with a qualified healthcare provider for medical advice
- Review ECG strips with a cardiologist for clinical interpretation

## License

MIT License
