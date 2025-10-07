-- Migration: Create tables for pipeline management
-- Created: 2025-10-06
-- SQLite Compatible

-- Pipeline table
CREATE TABLE IF NOT EXISTS pipeline (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Label configuration table
CREATE TABLE IF NOT EXISTS label_config (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    name TEXT NOT NULL,
    id2label TEXT NOT NULL, -- JSON: {"0": "payment_request", "1": "payment_intent", "2": "open_intent"}
    label2id TEXT NOT NULL, -- JSON: {"payment_request": 0, "payment_intent": 1, "open_intent": 2}
    label_explanation TEXT, -- JSON: {"payment_intent": "description...", ...}
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline(id) ON DELETE CASCADE
);

-- Pipeline phase table
CREATE TABLE IF NOT EXISTS pipeline_phase (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    previous_phase_id TEXT,
    phase_number INTEGER NOT NULL,
    checkpoint_id TEXT,
    checkpoint_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, in_progress, completed, failed
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline(id) ON DELETE CASCADE,
    FOREIGN KEY (previous_phase_id) REFERENCES pipeline_phase(id) ON DELETE SET NULL
);

-- Composal dataset table
CREATE TABLE IF NOT EXISTS composal_dataset (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    name TEXT,
    description TEXT,
    total_samples INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline(id) ON DELETE CASCADE
);

-- Dataset file table
CREATE TABLE IF NOT EXISTS dataset_file (
    id TEXT PRIMARY KEY,
    parent_dataset_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    phase_id TEXT NOT NULL,
    file_type TEXT, -- train, validation, test, generated
    sample_count INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_dataset_id) REFERENCES composal_dataset(id) ON DELETE CASCADE,
    FOREIGN KEY (phase_id) REFERENCES pipeline_phase(id) ON DELETE CASCADE
);

-- Error bucket table
CREATE TABLE IF NOT EXISTS error_bucket (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    examples TEXT NOT NULL, -- JSON serialized list of Sample objects
    data_generation_strategy TEXT, -- Strategy for generating data for this bucket
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline(id) ON DELETE CASCADE
);

-- Phase error bucket table
CREATE TABLE IF NOT EXISTS phase_error_bucket (
    id TEXT PRIMARY KEY,
    phase_id TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    error_count INTEGER NOT NULL DEFAULT 0,
    generation_count INTEGER NOT NULL DEFAULT 0,
    examples TEXT NOT NULL, -- JSON serialized list of Sample objects
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (phase_id) REFERENCES pipeline_phase(id) ON DELETE CASCADE,
    FOREIGN KEY (bucket_id) REFERENCES error_bucket(id) ON DELETE CASCADE
);

-- Create indexes for foreign keys
CREATE INDEX IF NOT EXISTS idx_label_config_pipeline ON label_config(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_phase_pipeline ON pipeline_phase(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_composal_dataset_pipeline ON composal_dataset(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_dataset_file_parent ON dataset_file(parent_dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_file_phase ON dataset_file(phase_id);
CREATE INDEX IF NOT EXISTS idx_error_bucket_pipeline ON error_bucket(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_phase_error_bucket_phase ON phase_error_bucket(phase_id);
CREATE INDEX IF NOT EXISTS idx_phase_error_bucket_bucket ON phase_error_bucket(bucket_id);
