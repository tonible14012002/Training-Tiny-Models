-- Migration: Add training and evaluation tables
-- Created: 2025-10-07
-- SQLite Compatible

-- Human test set table for storing custom human dataset files
CREATE TABLE IF NOT EXISTS human_test_set (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    description TEXT,
    sample_count INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline(id) ON DELETE SET NULL
);

-- Trained model table for storing trained model information
CREATE TABLE IF NOT EXISTS trained_model (
    id TEXT PRIMARY KEY,
    phase_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_save_path TEXT NOT NULL,
    training_time REAL, -- Training time in seconds
    dataset_file_id TEXT,
    training_params TEXT, -- JSON string with training parameters
    status TEXT NOT NULL DEFAULT 'training', -- training, completed, failed
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (phase_id) REFERENCES pipeline_phase(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_file_id) REFERENCES dataset_file(id) ON DELETE SET NULL
);

-- Evaluation result table for storing model evaluation results
CREATE TABLE IF NOT EXISTS evaluation_result (
    id TEXT PRIMARY KEY,
    trained_model_id TEXT NOT NULL,
    human_test_set_id TEXT,
    dataset_file_id TEXT,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    metrics TEXT, -- JSON string for additional metrics
    evaluated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trained_model_id) REFERENCES trained_model(id) ON DELETE CASCADE,
    FOREIGN KEY (human_test_set_id) REFERENCES human_test_set(id) ON DELETE SET NULL,
    FOREIGN KEY (dataset_file_id) REFERENCES dataset_file(id) ON DELETE SET NULL
);

-- Create indexes for foreign keys
CREATE INDEX IF NOT EXISTS idx_human_test_set_pipeline ON human_test_set(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_trained_model_phase ON trained_model(phase_id);
CREATE INDEX IF NOT EXISTS idx_trained_model_dataset ON trained_model(dataset_file_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_result_model ON evaluation_result(trained_model_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_result_human_test ON evaluation_result(human_test_set_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_result_dataset ON evaluation_result(dataset_file_id);
