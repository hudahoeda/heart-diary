-- Heart Diary Database Migration: Initial Schema (SQLite version)
-- Version: 001
-- Description: Create initial tables with normalized structure
-- Date: 2025-12-03

-- ============================================
-- 1. Reports Table (Main metadata)
-- ============================================
CREATE TABLE IF NOT EXISTS reports (
    id VARCHAR(8) PRIMARY KEY,
    date TIMESTAMP NOT NULL,
    duration_min FLOAT DEFAULT 0,
    mean_hr INTEGER DEFAULT 0,
    max_hr INTEGER DEFAULT 0,
    min_hr INTEGER DEFAULT 0,
    ectopic_burden FLOAT DEFAULT 0,
    sdnn INTEGER DEFAULT 0,
    pnn50 INTEGER DEFAULT 0,
    notes TEXT,
    ecg_filename VARCHAR(255),
    has_acc BOOLEAN DEFAULT FALSE,
    has_marker BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reports_date ON reports(date);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);

-- ============================================
-- 2. ECG Data Table
-- ============================================
CREATE TABLE IF NOT EXISTS ecg_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id VARCHAR(8) NOT NULL UNIQUE REFERENCES reports(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    content TEXT NOT NULL,
    sampling_rate INTEGER,
    sample_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ecg_data_report_id ON ecg_data(report_id);

-- ============================================
-- 3. Accelerometer Data Table
-- ============================================
CREATE TABLE IF NOT EXISTS acc_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id VARCHAR(8) NOT NULL UNIQUE REFERENCES reports(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    content TEXT NOT NULL,
    rest_ratio FLOAT DEFAULT 100.0,
    active_ratio FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_acc_data_report_id ON acc_data(report_id);

-- ============================================
-- 4. Marker Data Table (one row per marker)
-- ============================================
CREATE TABLE IF NOT EXISTS marker_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id VARCHAR(8) NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    marker_time TIMESTAMP,
    marker_label VARCHAR(255),
    sample_index INTEGER,
    
    -- Pattern detection
    detected_pattern VARCHAR(100),
    pattern_severity INTEGER DEFAULT 0,
    hr_at_marker INTEGER,
    
    -- ECG plot around marker (base64)
    ecg_plot_base64 TEXT,
    
    -- Raw content (stored in first marker only)
    raw_content TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_marker_data_report_id ON marker_data(report_id);

-- ============================================
-- 5. Report HTML Table
-- ============================================
CREATE TABLE IF NOT EXISTS report_html (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id VARCHAR(8) NOT NULL UNIQUE REFERENCES reports(id) ON DELETE CASCADE,
    html_content TEXT NOT NULL,
    baseline_plot_base64 TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_report_html_report_id ON report_html(report_id);
