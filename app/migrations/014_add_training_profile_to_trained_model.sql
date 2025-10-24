-- Migration: Add training_argument_profile_id to trained_model table
-- Description: Add foreign key reference to training_argument_profile for tracking which profile was used
-- Created: 2025-10-24

-- Add training_argument_profile_id column (nullable)
ALTER TABLE trained_model ADD COLUMN training_argument_profile_id TEXT;

-- Create foreign key index
CREATE INDEX IF NOT EXISTS idx_trained_model_training_profile ON trained_model(training_argument_profile_id);

-- Note: SQLite doesn't support adding foreign key constraints to existing tables,
-- but the application layer will enforce the relationship through the ORM
