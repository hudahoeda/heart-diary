"""
Heart Diary - FastAPI Application for Polar H10 ECG Analysis
"""
import os
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ecg_processor import process_ecg_data, get_report_html

# Initialize FastAPI app
app = FastAPI(title="Heart Diary", description="Polar H10 ECG Analysis Dashboard")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/heartdiary.db")

# Handle SQLite vs PostgreSQL
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Model
class Report(Base):
    __tablename__ = "reports"
    
    id = Column(String(8), primary_key=True)
    date = Column(DateTime, nullable=False)
    duration_min = Column(Float, default=0)
    mean_hr = Column(Integer, default=0)
    max_hr = Column(Integer, default=0)
    min_hr = Column(Integer, default=0)
    ectopic_burden = Column(Float, default=0)
    sdnn = Column(Integer, default=0)
    pnn50 = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    ecg_filename = Column(String(255), nullable=True)
    has_acc = Column(Boolean, default=False)
    has_marker = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)

# Create directories
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

for directory in [UPLOADS_DIR, REPORTS_DIR, TEMPLATES_DIR, STATIC_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_reports_db() -> dict:
    """Load reports from database."""
    db = SessionLocal()
    try:
        reports = db.query(Report).order_by(Report.date.desc()).all()
        return {
            "reports": [
                {
                    "id": r.id,
                    "date": r.date.isoformat() if r.date else None,
                    "duration_min": r.duration_min,
                    "mean_hr": r.mean_hr,
                    "max_hr": r.max_hr,
                    "ectopic_burden": r.ectopic_burden,
                    "notes": r.notes,
                    "ecg_filename": r.ecg_filename,
                    "has_acc": r.has_acc,
                    "has_marker": r.has_marker,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in reports
            ]
        }
    finally:
        db.close()


def save_report(report_data: dict):
    """Save a report to database."""
    db = SessionLocal()
    try:
        report = Report(
            id=report_data["id"],
            date=datetime.fromisoformat(report_data["date"]) if report_data.get("date") else datetime.utcnow(),
            duration_min=report_data.get("duration_min", 0),
            mean_hr=report_data.get("mean_hr", 0),
            max_hr=report_data.get("max_hr", 0),
            ectopic_burden=report_data.get("ectopic_burden", 0),
            notes=report_data.get("notes", ""),
            ecg_filename=report_data.get("ecg_filename", ""),
            has_acc=report_data.get("has_acc", False),
            has_marker=report_data.get("has_marker", False),
            created_at=datetime.utcnow()
        )
        db.add(report)
        db.commit()
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/calendar", response_class=HTMLResponse)
async def calendar_view(request: Request):
    """Calendar view showing all reports."""
    db = load_reports_db()
    reports = db.get("reports", [])
    
    # Organize reports by date for calendar
    reports_by_date = {}
    for report in reports:
        date = report.get("date", "")[:10]  # Get YYYY-MM-DD
        if date not in reports_by_date:
            reports_by_date[date] = []
        reports_by_date[date].append(report)
    
    return templates.TemplateResponse("calendar.html", {
        "request": request,
        "reports": reports,
        "reports_by_date": json.dumps(reports_by_date)
    })


@app.get("/report/{report_id}", response_class=HTMLResponse)
async def view_report(request: Request, report_id: str):
    """View a specific report."""
    db = load_reports_db()
    report = None
    for r in db.get("reports", []):
        if r.get("id") == report_id:
            report = r
            break
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Read the report HTML
    report_path = REPORTS_DIR / f"{report_id}.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    with open(report_path, "r") as f:
        report_html = f.read()
    
    return HTMLResponse(content=report_html)


@app.post("/upload")
async def upload_files(
    request: Request,
    ecg_file: UploadFile = File(...),
    acc_file: Optional[UploadFile] = File(None),
    marker_file: Optional[UploadFile] = File(None),
    notes: str = Form("")
):
    """Upload and process ECG files."""
    report_id = str(uuid.uuid4())[:8]
    upload_subdir = UPLOADS_DIR / report_id
    upload_subdir.mkdir(exist_ok=True)
    
    try:
        # Save ECG file
        ecg_path = upload_subdir / ecg_file.filename
        with open(ecg_path, "wb") as f:
            content = await ecg_file.read()
            f.write(content)
        
        # Save ACC file if provided
        acc_path = None
        if acc_file and acc_file.filename:
            acc_path = upload_subdir / acc_file.filename
            with open(acc_path, "wb") as f:
                content = await acc_file.read()
                f.write(content)
        
        # Save Marker file if provided
        marker_path = None
        if marker_file and marker_file.filename:
            marker_path = upload_subdir / marker_file.filename
            with open(marker_path, "wb") as f:
                content = await marker_file.read()
                f.write(content)
        
        # Process the data
        result = process_ecg_data(
            str(ecg_path),
            str(marker_path) if marker_path else None,
            str(acc_path) if acc_path else None,
            str(REPORTS_DIR),
            report_id
        )
        
        if result.get("success"):
            # Save report to database
            report_entry = {
                "id": report_id,
                "date": result.get("date", datetime.now().isoformat()),
                "duration_min": result.get("duration_min", 0),
                "mean_hr": result.get("mean_hr", 0),
                "max_hr": result.get("max_hr", 0),
                "ectopic_burden": result.get("ectopic_burden", 0),
                "notes": notes,
                "ecg_filename": ecg_file.filename,
                "has_acc": acc_path is not None,
                "has_marker": marker_path is not None
            }
            save_report(report_entry)
            
            return RedirectResponse(url=f"/report/{report_id}", status_code=303)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Processing failed"))
            
    except Exception as e:
        # Clean up on error
        if upload_subdir.exists():
            shutil.rmtree(upload_subdir)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/report/{report_id}")
async def delete_report(report_id: str):
    """Delete a report."""
    db = SessionLocal()
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if report:
            db.delete(report)
            db.commit()
    finally:
        db.close()
    
    # Delete report file
    report_path = REPORTS_DIR / f"{report_id}.html"
    if report_path.exists():
        report_path.unlink()
    
    # Delete uploads
    upload_dir = UPLOADS_DIR / report_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    
    return {"status": "deleted"}


@app.get("/api/reports")
async def get_reports():
    """Get all reports as JSON (for calendar)."""
    db = load_reports_db()
    return db.get("reports", [])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
