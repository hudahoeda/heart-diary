"""
Heart Diary - FastAPI Application for Polar H10 ECG Analysis
"""
import os
import secrets
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import re

from ecg_processor import process_ecg_data

# Import database models and connection from database package
from database import (
    Base, Report, ECGData, AccelerometerData, HRData, MarkerData, ReportHTML,
    SessionLocal, get_db
)
from database.connection import engine
from database.migrate import create_tables

# Initialize FastAPI app
app = FastAPI(title="Heart Diary", description="Polar H10 ECG Analysis Dashboard")

# HTTP Basic Auth
security = HTTPBasic()

AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "changeme")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic Auth credentials."""
    correct_username = secrets.compare_digest(credentials.username, AUTH_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Create database tables on startup
create_tables()

# Create directories (only static and data needed now)
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

for directory in [TEMPLATES_DIR, STATIC_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files (only static needed, reports are served from database)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def load_reports_db() -> dict:
    """Load reports from database with related data."""
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
                    "min_hr": r.min_hr,
                    "ectopic_burden": r.ectopic_burden,
                    "sdnn": r.sdnn,
                    "pnn50": r.pnn50,
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


def save_report_to_db(report_data: dict, ecg_content: str, acc_content: str = None, 
                      marker_content: str = None, hr_content: str = None, markers_list: list = None):
    """
    Save a report to database with separate tables for each data type.
    """
    db = SessionLocal()
    try:
        # 1. Create main report record
        report = Report(
            id=report_data["id"],
            date=datetime.fromisoformat(report_data["date"]) if report_data.get("date") else datetime.utcnow(),
            duration_min=report_data.get("duration_min", 0),
            mean_hr=report_data.get("mean_hr", 0),
            max_hr=report_data.get("max_hr", 0),
            min_hr=report_data.get("min_hr", 0),
            ectopic_burden=report_data.get("ectopic_burden", 0),
            sdnn=report_data.get("sdnn", 0),
            pnn50=report_data.get("pnn50", 0),
            notes=report_data.get("notes", ""),
            ecg_filename=report_data.get("ecg_filename", ""),
            has_acc=acc_content is not None,
            has_marker=marker_content is not None,
            has_hr=hr_content is not None,
            created_at=datetime.utcnow()
        )
        db.add(report)
        db.flush()  # Get the ID before adding related records
        
        # 2. Create ECG data record
        ecg_record = ECGData(
            report_id=report.id,
            filename=report_data.get("ecg_filename"),
            content=ecg_content,
            sampling_rate=report_data.get("sampling_rate"),
            sample_count=report_data.get("sample_count")
        )
        db.add(ecg_record)
        
        # 3. Create Accelerometer data record (if provided)
        if acc_content:
            acc_record = AccelerometerData(
                report_id=report.id,
                filename=report_data.get("acc_filename"),
                content=acc_content,
                rest_ratio=report_data.get("motion_stats", {}).get("rest_ratio", 100.0),
                active_ratio=report_data.get("motion_stats", {}).get("active_ratio", 0.0)
            )
            db.add(acc_record)
        
        # 4. Create HR data record (if provided)
        if hr_content:
            hr_record = HRData(
                report_id=report.id,
                filename=report_data.get("hr_filename"),
                content=hr_content
            )
            db.add(hr_record)
        
        # 5. Create Marker data records (if provided)
        if markers_list:
            for i, marker in enumerate(markers_list):
                marker_record = MarkerData(
                    report_id=report.id,
                    marker_time=datetime.fromisoformat(marker["marker_time"]) if marker.get("marker_time") else None,
                    marker_label=marker.get("marker_label"),
                    sample_index=marker.get("sample_index"),
                    detected_pattern=marker.get("detected_pattern"),
                    pattern_severity=marker.get("pattern_severity", 0),
                    hr_at_marker=marker.get("hr_at_marker"),
                    ecg_plot_base64=marker.get("ecg_plot_base64"),
                    raw_content=marker_content if i == 0 else None  # Store raw content only in first record
                )
                db.add(marker_record)
        
        # 6. Create Report HTML record
        if report_data.get("report_html"):
            html_record = ReportHTML(
                report_id=report.id,
                html_content=report_data["report_html"],
                baseline_plot_base64=report_data.get("baseline_plot_base64")
            )
            db.add(html_record)
        
        db.commit()
        return report.id
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration (no auth required)."""
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, username: str = Depends(verify_credentials)):
    """Home page with upload form."""
    return templates.TemplateResponse("app.html", {
        "request": request,
        "active_tab": "upload",
        "page_title": "Upload ECG"
    })


@app.get("/calendar", response_class=HTMLResponse)
async def calendar_view(request: Request, username: str = Depends(verify_credentials)):
    """Calendar view showing all reports."""
    db_data = load_reports_db()
    reports = db_data.get("reports", [])
    
    # Organize reports by date for calendar
    reports_by_date = {}
    for report in reports:
        date = report.get("date", "")[:10]  # Get YYYY-MM-DD
        if date not in reports_by_date:
            reports_by_date[date] = []
        reports_by_date[date].append(report)
    
    return templates.TemplateResponse("app.html", {
        "request": request,
        "active_tab": "calendar",
        "page_title": "Calendar",
        "reports": reports,
        "reports_by_date": json.dumps(reports_by_date)
    })


@app.get("/report/{report_id}", response_class=HTMLResponse)
async def view_report(request: Request, report_id: str, username: str = Depends(verify_credentials)):
    """View a specific report - served from database."""
    db = SessionLocal()
    try:
        # Get report HTML from separate table
        report_html = db.query(ReportHTML).filter(ReportHTML.report_id == report_id).first()
        if not report_html:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return HTMLResponse(content=report_html.html_content)
    finally:
        db.close()


@app.post("/upload")
async def upload_files(
    request: Request,
    ecg_file: UploadFile = File(...),
    acc_file: Optional[UploadFile] = File(None),
    marker_file: Optional[UploadFile] = File(None),
    hr_file: Optional[UploadFile] = File(None),
    notes: str = Form(""),
    username: str = Depends(verify_credentials)
):
    """Upload and process ECG files - store in database instead of files."""
    report_id = str(uuid.uuid4())[:8]
    
    try:
        # Read file contents as strings (decode from bytes)
        ecg_content = (await ecg_file.read()).decode('utf-8')
        
        acc_content = None
        if acc_file and acc_file.filename:
            acc_content = (await acc_file.read()).decode('utf-8')
        
        marker_content = None
        if marker_file and marker_file.filename:
            marker_content = (await marker_file.read()).decode('utf-8')
        
        hr_content = None
        if hr_file and hr_file.filename:
            hr_content = (await hr_file.read()).decode('utf-8')
        
        # Process the data using content strings (no file I/O)
        result = process_ecg_data(
            ecg_content,
            marker_content,
            acc_content,
            hr_content,
            report_id
        )
        
        if result.get("success"):
            # Save report to database with file contents and generated HTML
            report_entry = {
                "id": report_id,
                "date": result.get("date", datetime.now().isoformat()),
                "duration_min": result.get("duration_min", 0),
                "mean_hr": result.get("mean_hr", 0),
                "max_hr": result.get("max_hr", 0),
                "min_hr": result.get("min_hr", 0),
                "ectopic_burden": result.get("ectopic_burden", 0),
                "sdnn": result.get("sdnn", 0),
                "pnn50": result.get("pnn50", 0),
                "notes": notes,
                "ecg_filename": ecg_file.filename,
                "acc_filename": acc_file.filename if acc_file and acc_file.filename else None,
                "hr_filename": hr_file.filename if hr_file and hr_file.filename else None,
                "sampling_rate": result.get("sampling_rate"),
                "sample_count": result.get("sample_count"),
                "motion_stats": result.get("motion_stats", {}),
                "report_html": result.get("report_html"),
                "baseline_plot_base64": result.get("baseline_plot_base64")
            }
            
            # Save to database with separate tables
            save_report_to_db(
                report_entry,
                ecg_content=ecg_content,
                acc_content=acc_content,
                marker_content=marker_content,
                hr_content=hr_content,
                markers_list=result.get("markers", [])
            )
            
            return RedirectResponse(url=f"/report/{report_id}", status_code=303)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Processing failed"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/report/{report_id}")
async def delete_report(report_id: str, username: str = Depends(verify_credentials)):
    """Delete a report and all related data from database (cascade delete)."""
    db = SessionLocal()
    try:
        report = db.query(Report).filter(Report.id == report_id).first()
        if report:
            # Cascade delete will remove related records from other tables
            db.delete(report)
            db.commit()
            return {"status": "deleted"}
        else:
            raise HTTPException(status_code=404, detail="Report not found")
    finally:
        db.close()


@app.post("/report/{report_id}/recalculate")
async def recalculate_report(report_id: str, username: str = Depends(verify_credentials)):
    """
    Recalculate a report using stored raw data from the database.
    This regenerates the HTML report with any algorithm updates.
    """
    db = SessionLocal()
    try:
        # 1. Get the report and related data
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # 2. Get ECG data (required)
        ecg_data = db.query(ECGData).filter(ECGData.report_id == report_id).first()
        if not ecg_data:
            raise HTTPException(status_code=404, detail="ECG data not found for this report")
        
        ecg_content = ecg_data.content
        
        # 3. Get optional data
        acc_content = None
        acc_data = db.query(AccelerometerData).filter(AccelerometerData.report_id == report_id).first()
        if acc_data:
            acc_content = acc_data.content
        
        hr_content = None
        hr_data = db.query(HRData).filter(HRData.report_id == report_id).first()
        if hr_data:
            hr_content = hr_data.content
        
        marker_content = None
        marker_data = db.query(MarkerData).filter(MarkerData.report_id == report_id).first()
        if marker_data and marker_data.raw_content:
            marker_content = marker_data.raw_content
        
        # 4. Reprocess the data
        result = process_ecg_data(
            ecg_content,
            marker_content,
            acc_content,
            hr_content,
            report_id
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Recalculation failed"))
        
        # 5. Update the report record with new stats
        report.duration_min = result.get("duration_min", report.duration_min)
        report.mean_hr = result.get("mean_hr", report.mean_hr)
        report.max_hr = result.get("max_hr", report.max_hr)
        report.min_hr = result.get("min_hr", report.min_hr)
        report.ectopic_burden = result.get("ectopic_burden", report.ectopic_burden)
        report.sdnn = result.get("sdnn", report.sdnn)
        report.pnn50 = result.get("pnn50", report.pnn50)
        
        # 6. Update the HTML report
        report_html_record = db.query(ReportHTML).filter(ReportHTML.report_id == report_id).first()
        if report_html_record:
            report_html_record.html_content = result.get("report_html", report_html_record.html_content)
            report_html_record.baseline_plot_base64 = result.get("baseline_plot_base64", report_html_record.baseline_plot_base64)
            report_html_record.generated_at = datetime.utcnow()
        elif result.get("report_html"):
            # Create new HTML record if it doesn't exist
            new_html_record = ReportHTML(
                report_id=report_id,
                html_content=result["report_html"],
                baseline_plot_base64=result.get("baseline_plot_base64")
            )
            db.add(new_html_record)
        
        # 7. Update marker records with new pattern detection
        if result.get("markers"):
            # Delete old markers and add new ones
            db.query(MarkerData).filter(MarkerData.report_id == report_id).delete()
            for i, marker in enumerate(result["markers"]):
                marker_record = MarkerData(
                    report_id=report_id,
                    marker_time=datetime.fromisoformat(marker["marker_time"]) if marker.get("marker_time") else None,
                    marker_label=marker.get("marker_label"),
                    sample_index=marker.get("sample_index"),
                    detected_pattern=marker.get("detected_pattern"),
                    pattern_severity=marker.get("pattern_severity", 0),
                    hr_at_marker=marker.get("hr_at_marker"),
                    ecg_plot_base64=marker.get("ecg_plot_base64"),
                    raw_content=marker_content if i == 0 else None
                )
                db.add(marker_record)
        
        db.commit()
        
        return RedirectResponse(url=f"/report/{report_id}", status_code=303)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


def match_polar_files(files: list) -> list:
    """
    Match Polar files into groups based on device ID and timestamp.
    Returns a list of dicts with matched ECG, HR, ACC, Marker files.
    """
    # Pattern: Polar_H10_<DEVICE_ID>_<DATE>_<TIME>_<TYPE>.txt
    # Example: Polar_H10_EC4EFF22_20251203_135206_ECG.txt
    pattern = r'Polar_H10_([A-F0-9]+)_(\d{8})_(\d{6})_([A-Z]+)\.txt'
    
    file_groups = {}  # key: (device_id, date, approx_time) -> files dict
    
    for file in files:
        match = re.match(pattern, file.filename, re.IGNORECASE)
        if match:
            device_id, date_str, time_str, file_type = match.groups()
            file_type = file_type.upper()
            
            # Group by device and date, using time window (within 30 seconds)
            time_int = int(time_str)
            
            # Find or create group
            group_key = None
            for key in file_groups.keys():
                if key[0] == device_id and key[1] == date_str:
                    # Check if time is within 60 seconds
                    existing_time = int(key[2])
                    if abs(time_int - existing_time) <= 100:  # ~1 minute tolerance in HHMMSS
                        group_key = key
                        break
            
            if group_key is None:
                group_key = (device_id, date_str, time_str)
                file_groups[group_key] = {'ecg': None, 'hr': None, 'acc': None, 'marker': None}
            
            # Assign file to appropriate slot
            if file_type == 'ECG':
                file_groups[group_key]['ecg'] = file
            elif file_type == 'HR':
                file_groups[group_key]['hr'] = file
            elif file_type == 'ACC':
                file_groups[group_key]['acc'] = file
            elif file_type == 'RR':
                # RR files can be used if HR is not available (optional)
                if file_groups[group_key]['hr'] is None:
                    pass  # Could add RR support later
    
    # Also check for marker files (different pattern)
    marker_pattern = r'MARKER_(\d{8})_(\d{6})\.txt'
    for file in files:
        match = re.match(marker_pattern, file.filename, re.IGNORECASE)
        if match:
            date_str, time_str = match.groups()
            time_int = int(time_str)
            
            # Try to match with existing groups
            for key, group in file_groups.items():
                if key[1] == date_str:
                    existing_time = int(key[2])
                    if abs(time_int - existing_time) <= 1000:  # ~10 minute tolerance
                        group['marker'] = file
                        break
    
    # Filter to only groups with ECG files
    valid_groups = [
        {
            'device_id': key[0],
            'date': key[1],
            'time': key[2],
            **group
        }
        for key, group in file_groups.items()
        if group['ecg'] is not None
    ]
    
    return valid_groups


@app.post("/bulk-upload")
async def bulk_upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    notes: str = Form(""),
    username: str = Depends(verify_credentials)
):
    """
    Bulk upload multiple Polar files. Automatically matches ECG, HR, ACC, and Marker files.
    Returns JSON with results for each processed session.
    """
    results = []
    
    # Match files into groups
    file_groups = match_polar_files(files)
    
    if not file_groups:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid ECG files found. Files must be named like: Polar_H10_DEVICEID_YYYYMMDD_HHMMSS_ECG.txt"}
        )
    
    for group in file_groups:
        report_id = str(uuid.uuid4())[:8]
        session_time = f"{group['date']}_{group['time']}"
        
        try:
            # Read file contents
            ecg_content = (await group['ecg'].read()).decode('utf-8')
            await group['ecg'].seek(0)  # Reset for potential re-read
            
            hr_content = None
            if group['hr']:
                hr_content = (await group['hr'].read()).decode('utf-8')
                await group['hr'].seek(0)
            
            acc_content = None
            if group['acc']:
                acc_content = (await group['acc'].read()).decode('utf-8')
                await group['acc'].seek(0)
            
            marker_content = None
            if group['marker']:
                marker_content = (await group['marker'].read()).decode('utf-8')
                await group['marker'].seek(0)
            
            # Process the data
            result = process_ecg_data(
                ecg_content,
                marker_content,
                acc_content,
                hr_content,
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
                    "min_hr": result.get("min_hr", 0),
                    "ectopic_burden": result.get("ectopic_burden", 0),
                    "sdnn": result.get("sdnn", 0),
                    "pnn50": result.get("pnn50", 0),
                    "notes": notes,
                    "ecg_filename": group['ecg'].filename,
                    "acc_filename": group['acc'].filename if group['acc'] else None,
                    "sampling_rate": result.get("sampling_rate"),
                    "sample_count": result.get("sample_count"),
                    "motion_stats": result.get("motion_stats", {}),
                    "report_html": result.get("report_html"),
                    "baseline_plot_base64": result.get("baseline_plot_base64")
                }
                
                save_report_to_db(
                    report_entry,
                    ecg_content=ecg_content,
                    acc_content=acc_content,
                    marker_content=marker_content,
                    markers_list=result.get("markers", [])
                )
                
                results.append({
                    "status": "success",
                    "report_id": report_id,
                    "session": session_time,
                    "ecg_file": group['ecg'].filename,
                    "hr_file": group['hr'].filename if group['hr'] else None,
                    "acc_file": group['acc'].filename if group['acc'] else None,
                    "marker_file": group['marker'].filename if group['marker'] else None,
                    "mean_hr": result.get("mean_hr"),
                    "duration_min": result.get("duration_min")
                })
            else:
                results.append({
                    "status": "error",
                    "session": session_time,
                    "ecg_file": group['ecg'].filename,
                    "error": result.get("error", "Processing failed")
                })
                
        except Exception as e:
            results.append({
                "status": "error",
                "session": session_time,
                "ecg_file": group['ecg'].filename if group.get('ecg') else "unknown",
                "error": str(e)
            })
    
    # If only one successful result, redirect to the report
    successful = [r for r in results if r["status"] == "success"]
    if len(successful) == 1:
        return RedirectResponse(url=f"/report/{successful[0]['report_id']}", status_code=303)
    
    # Otherwise redirect to bulk results page
    return templates.TemplateResponse("app.html", {
        "request": request,
        "active_tab": "bulk_results",
        "page_title": "Bulk Upload Results",
        "bulk_results": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "results": results
        }
    })


@app.get("/api/reports")
async def get_reports(username: str = Depends(verify_credentials)):
    """Get all reports as JSON (for calendar)."""
    db_data = load_reports_db()
    return db_data.get("reports", [])


@app.get("/api/report/{report_id}/markers")
async def get_report_markers(report_id: str, username: str = Depends(verify_credentials)):
    """Get all markers for a specific report with pattern data."""
    db = SessionLocal()
    try:
        markers = db.query(MarkerData).filter(MarkerData.report_id == report_id).all()
        return [
            {
                "id": m.id,
                "marker_time": m.marker_time.isoformat() if m.marker_time else None,
                "marker_label": m.marker_label,
                "sample_index": m.sample_index,
                "detected_pattern": m.detected_pattern,
                "pattern_severity": m.pattern_severity,
                "hr_at_marker": m.hr_at_marker
            }
            for m in markers
        ]
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
