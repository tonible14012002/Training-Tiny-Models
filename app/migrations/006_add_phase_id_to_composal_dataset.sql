-- Migration: Add phase_id to composal_dataset table
-- Created: 2025-10-13
-- SQLite Compatible

-- Add phase_id column to track which phase generated this composal dataset
-- Each first-gen API call creates a new phase with a composal dataset
ALTER TABLE composal_dataset ADD COLUMN phase_id TEXT;

-- Add foreign key constraint via index (SQLite doesn't support adding FK to existing table)
-- The phase_id should reference pipeline_phase.id
CREATE INDEX IF NOT EXISTS idx_composal_dataset_phase ON composal_dataset(phase_id);

-- Note: In SQLite, we cannot add a foreign key constraint to an existing table
-- The foreign key relationship should be enforced at the application level
-- New tables should include: FOREIGN KEY (phase_id) REFERENCES pipeline_phase(id) ON DELETE SET NULL
