"""
SQLAlchemy ORM Models for Heart Diary
Each data type has its own table with foreign key relationships.
"""
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Report(Base):
    """
    Main report metadata table.
    Contains summary statistics and links to related data tables.
    """
    __tablename__ = "reports"
    
    id = Column(String(8), primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
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
    has_hr = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships to related data
    ecg_data = relationship("ECGData", back_populates="report", uselist=False, cascade="all, delete-orphan")
    acc_data = relationship("AccelerometerData", back_populates="report", uselist=False, cascade="all, delete-orphan")
    hr_data = relationship("HRData", back_populates="report", uselist=False, cascade="all, delete-orphan")
    markers = relationship("MarkerData", back_populates="report", cascade="all, delete-orphan")
    report_html = relationship("ReportHTML", back_populates="report", uselist=False, cascade="all, delete-orphan")


class ECGData(Base):
    """
    ECG raw data storage table.
    Stores the original ECG file content.
    """
    __tablename__ = "ecg_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(8), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, unique=True)
    filename = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)  # Raw CSV/TXT content
    sampling_rate = Column(Integer, nullable=True)
    sample_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to report
    report = relationship("Report", back_populates="ecg_data")


class AccelerometerData(Base):
    """
    Accelerometer raw data storage table.
    Stores the original accelerometer file content.
    """
    __tablename__ = "acc_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(8), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, unique=True)
    filename = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)  # Raw CSV/TXT content
    rest_ratio = Column(Float, default=100.0)
    active_ratio = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to report
    report = relationship("Report", back_populates="acc_data")


class HRData(Base):
    """
    Polar HR file data storage table.
    Stores the original HR file content for recalculation.
    """
    __tablename__ = "hr_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(8), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, unique=True)
    filename = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)  # Raw CSV/TXT content
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to report
    report = relationship("Report", back_populates="hr_data")


class MarkerData(Base):
    """
    Marker/symptom data storage table.
    Each marker is stored as a separate row with detected patterns.
    """
    __tablename__ = "marker_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(8), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, index=True)
    marker_time = Column(DateTime, nullable=True)  # Timestamp of the marker
    marker_label = Column(String(255), nullable=True)  # User-provided label
    sample_index = Column(Integer, nullable=True)  # Index in ECG signal
    
    # Pattern detection around marker
    detected_pattern = Column(String(100), nullable=True)  # e.g., "Couplet", "Bigeminy", "Normal"
    pattern_severity = Column(Integer, default=0)  # 0=none, 1=mild, 2=moderate, 3=severe
    hr_at_marker = Column(Integer, nullable=True)  # Heart rate at marker time
    
    # Base64 encoded ECG plot around marker
    ecg_plot_base64 = Column(Text, nullable=True)
    
    # Raw marker file content (stored once per report, in first marker row)
    raw_content = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to report
    report = relationship("Report", back_populates="markers")


class ReportHTML(Base):
    """
    Generated HTML report storage table.
    Contains the full rendered report with embedded base64 images.
    """
    __tablename__ = "report_html"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(8), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, unique=True)
    html_content = Column(Text, nullable=False)  # Full HTML report
    
    # Separate storage for key images (base64)
    baseline_plot_base64 = Column(Text, nullable=True)  # Baseline ECG strip
    
    # Report generation metadata
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship back to report
    report = relationship("Report", back_populates="report_html")
