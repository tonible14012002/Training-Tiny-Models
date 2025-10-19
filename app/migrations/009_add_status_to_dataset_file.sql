-- Migration: Add status column to dataset_file table
-- Created: 2025-10-15
-- SQLite Compatible
--
-- This migration adds a status column to track the generation status of dataset files.
-- The status field can be 'generating' (during data generation) or 'done' (completed).

-- Add status column to dataset_file table
ALTER TABLE dataset_file ADD COLUMN status TEXT NOT NULL DEFAULT 'generating';

-- Create index for efficient status filtering
CREATE INDEX IF NOT EXISTS idx_dataset_file_status ON dataset_file(status);

-- Verification query (commented out for migration file):
-- SELECT id, file_path, status FROM dataset_file;
