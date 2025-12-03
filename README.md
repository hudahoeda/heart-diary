# Heart Diary 💓

A FastAPI web application for analyzing Polar H10 ECG data and tracking heart health over time.

## Features

- **Upload ECG Data**: Upload Polar H10 ECG logs, accelerometer data, and symptom markers
- **Automated Analysis**: Generates clinical-style ECG reports with:
  - Heart rate statistics (mean, max, min, HRV)
  - Ectopic burden calculation
  - Rhythm event detection (couplets, bigeminy, trigeminy)
  - Pattern detection around symptom markers
  - Baseline morphology (QRS, QTc)
  - Motion/activity detection from accelerometer data
  - Symptom marker visualization with pattern analysis
- **Calendar View**: Browse historical ECG recordings by date
- **Responsive Design**: Works on desktop and mobile devices
- **Docker Support**: Easy deployment with Docker Compose

## Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/hudahoeda/heart-diary.git
cd heart-diary

# Start with Docker Compose
docker compose up --build -d

# Access the app at http://localhost:8000
```

## Manual Installation

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

3. **Run the application**:
   ```bash
   # Development mode with auto-reload
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

   # Or run directly
   python main.py
   ```

Then open your browser to: **http://localhost:8000**

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | SQLite (dev) |
| `AUTH_USERNAME` | HTTP Basic Auth username | `admin` |
| `AUTH_PASSWORD` | HTTP Basic Auth password | `heartdiary` |

### Authentication

The app uses HTTP Basic Authentication. Set custom credentials via environment variables:

```bash
export AUTH_USERNAME=your_username
export AUTH_PASSWORD=your_secure_password
```

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
- Reports include pattern detection around symptom markers (couplets, bigeminy, trigeminy, etc.)

### File Formats

The app expects Polar H10 Sensor Logger export files:

- **ECG**: `Polar_H10_XXXXXX_YYYYMMDD_HHMMSS_ECG.txt`
- **Accelerometer**: `Polar_H10_XXXXXX_YYYYMMDD_HHMMSS_ACC.txt`
- **Markers**: `MARKER_YYYYMMDD_HHMMSS.txt`

## Project Structure

```
heart-diary/
├── main.py              # FastAPI application
├── ecg_processor.py     # ECG analysis module
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build configuration
├── docker-compose.yml   # Docker Compose setup
├── database/            # Database package
│   ├── __init__.py      # Package exports
│   ├── connection.py    # Database engine & sessions
│   ├── models.py        # SQLAlchemy ORM models
│   ├── migrate.py       # Migration utilities
│   └── migrations/      # SQL migration scripts
├── templates/
│   └── app.html         # Unified template
├── static/
│   └── style.css        # Stylesheets
└── data/
    └── heartdiary.db    # SQLite database (dev)
```

## Database Schema

The application uses a normalized database schema with separate tables:

| Table | Description |
|-------|-------------|
| `reports` | Report metadata (HR, duration, dates) |
| `ecg_data` | Raw ECG file content |
| `accelerometer_data` | Accelerometer file content |
| `marker_data` | Symptom markers with detected patterns |
| `report_html` | Generated HTML reports with embedded images |

- **Development**: SQLite (`data/heartdiary.db`)
- **Production**: PostgreSQL (via `DATABASE_URL`)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Upload form |
| `GET` | `/calendar` | Calendar view |
| `POST` | `/upload` | Process ECG upload |
| `GET` | `/report/{id}` | View report |
| `DELETE` | `/report/{id}` | Delete report |
| `GET` | `/api/reports` | List all reports (JSON) |
| `GET` | `/api/report/{id}/markers` | Get markers with patterns |

## Medical Disclaimer

⚠️ This application is for **informational purposes only** and is NOT a medical device. 

- Reports identify intervals consistent with premature beats but cannot definitively diagnose conditions
- QRS/QT values are estimates and may not be accurate
- Always consult with a qualified healthcare provider for medical advice
- Review ECG strips with a cardiologist for clinical interpretation

## License

MIT License
