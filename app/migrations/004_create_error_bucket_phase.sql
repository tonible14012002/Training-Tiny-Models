-- Migration: Create error_bucket_phase table
-- Created: 2025-10-07
-- SQLite Compatible

-- Error bucket phase table - stores phase-specific error bucket instances cloned from base ErrorBucket
CREATE TABLE IF NOT EXISTS error_bucket_phase (
    id TEXT PRIMARY KEY,
    phase_id TEXT NOT NULL,
    base_error_bucket_id TEXT NOT NULL, -- The base ErrorBucket this was cloned from
    name TEXT NOT NULL, -- Cloned from base bucket
    description TEXT NOT NULL, -- Cloned from base bucket
    examples TEXT NOT NULL, -- JSON list: [{"text": "...", "label": "...", "prediction": "..."}, ...]
    count INTEGER DEFAULT 0, -- Number of errors in this category for this phase
    data_generation_strategy TEXT, -- Cloned from base bucket
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (phase_id) REFERENCES pipeline_phase(id) ON DELETE CASCADE,
    FOREIGN KEY (base_error_bucket_id) REFERENCES error_bucket(id) ON DELETE CASCADE
);

-- Create indexes for foreign keys
CREATE INDEX IF NOT EXISTS idx_error_bucket_phase_phase ON error_bucket_phase(phase_id);
CREATE INDEX IF NOT EXISTS idx_error_bucket_phase_base_bucket ON error_bucket_phase(base_error_bucket_id);
