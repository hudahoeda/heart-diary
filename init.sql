-- Heart Diary Database Schema

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Store file contents in database
    ecg_data TEXT,           -- Raw ECG file content
    acc_data TEXT,           -- Raw accelerometer file content
    marker_data TEXT,        -- Raw marker file content
    report_html TEXT         -- Generated report HTML
);

-- Index for faster date queries
CREATE INDEX IF NOT EXISTS idx_reports_date ON reports(date);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);
