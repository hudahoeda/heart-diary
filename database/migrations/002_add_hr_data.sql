-- Heart Diary Database Migration: Add HR Data Support
-- Version: 002
-- Description: Add HR data table and has_hr column to reports
-- Date: 2025-12-03

-- ============================================
-- 1. Add has_hr column to reports table
-- ============================================
ALTER TABLE reports ADD COLUMN IF NOT EXISTS has_hr BOOLEAN DEFAULT FALSE;

-- ============================================
-- 2. HR Data Table (for Polar HR file storage)
-- ============================================
CREATE TABLE IF NOT EXISTS hr_data (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(8) NOT NULL UNIQUE REFERENCES reports(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hr_data_report_id ON hr_data(report_id);
