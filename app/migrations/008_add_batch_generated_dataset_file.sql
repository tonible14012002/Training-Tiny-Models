-- Migration: Add batch_generated_dataset_file table
-- Created: 2025-10-14
-- SQLite Compatible
--
-- This migration adds a new table to track batch-generated dataset files
-- during the iterative data generation process. Each batch represents a
-- subset of samples generated in a single iteration of the generation loop.

-- Create the batch_generated_dataset_file table
CREATE TABLE IF NOT EXISTS batch_generated_dataset_file (
    id TEXT PRIMARY KEY,
    parent_dataset_file_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    batch_number INTEGER NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_dataset_file_id) REFERENCES dataset_file(id) ON DELETE CASCADE
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_batch_dataset_file_parent ON batch_generated_dataset_file(parent_dataset_file_id);
CREATE INDEX IF NOT EXISTS idx_batch_dataset_file_batch_number ON batch_generated_dataset_file(batch_number);

-- Verification query (commented out for migration file):
-- SELECT * FROM batch_generated_dataset_file ORDER BY batch_number;
