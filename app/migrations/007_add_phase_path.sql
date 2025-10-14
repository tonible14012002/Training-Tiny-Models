-- Migration: Add phase_path column and remove previous_phase_id
-- Created: 2025-10-14
-- SQLite Compatible
--
-- This migration replaces the previous_phase_id foreign key with a phase_path column
-- that stores the hierarchical path of parent phases (e.g., "parent1_id/parent2_id")
-- This allows for better sequence tracking and grouping of phases by their first phase.

-- Step 1: Add the new phase_path column
ALTER TABLE pipeline_phase ADD COLUMN phase_path TEXT NOT NULL DEFAULT '';

-- Step 2: Populate phase_path for existing data
-- Set all phase_number = 0 phases to have empty string phase_path (base parent phases)
UPDATE pipeline_phase
SET phase_path = ''
WHERE phase_number = 0;

-- Step 3: For phases with previous_phase_id, build their phase_path
-- This is a multi-step process due to SQLite limitations with recursive CTEs in UPDATE statements
-- We'll need to handle this in the application code or do it iteratively

-- For phases with previous_phase_id that points to a phase_number = 0 phase (depth 1)
UPDATE pipeline_phase
SET phase_path = previous_phase_id
WHERE previous_phase_id IS NOT NULL
  AND previous_phase_id IN (SELECT id FROM pipeline_phase WHERE phase_number = 0);

-- For deeper levels (depth 2+), this needs to be done recursively in application code
-- The pattern is: phase_path = parent.phase_path + '/' + parent.id (if parent.phase_path is not empty)
--                           or parent.id (if parent.phase_path is empty)

-- Step 4: Create a temporary table to help with the recursive update
CREATE TEMPORARY TABLE temp_phase_paths AS
WITH RECURSIVE phase_hierarchy AS (
    -- Base case: phases with phase_number = 0 or no previous_phase
    SELECT
        id,
        previous_phase_id,
        phase_number,
        '' AS computed_path
    FROM pipeline_phase
    WHERE previous_phase_id IS NULL OR phase_number = 0

    UNION ALL

    -- Recursive case: build path from parent
    SELECT
        p.id,
        p.previous_phase_id,
        p.phase_number,
        CASE
            WHEN ph.computed_path = '' THEN ph.id
            ELSE ph.computed_path || '/' || ph.id
        END AS computed_path
    FROM pipeline_phase p
    INNER JOIN phase_hierarchy ph ON p.previous_phase_id = ph.id
)
SELECT id, computed_path FROM phase_hierarchy;

-- Step 5: Update phase_path using the computed values
UPDATE pipeline_phase
SET phase_path = (
    SELECT computed_path
    FROM temp_phase_paths
    WHERE temp_phase_paths.id = pipeline_phase.id
);

-- Step 6: Drop the temporary table
DROP TABLE temp_phase_paths;

-- Step 7: Create a new table without previous_phase_id
CREATE TABLE pipeline_phase_new (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    phase_path TEXT NOT NULL DEFAULT '',
    phase_number INTEGER NOT NULL,
    checkpoint_id TEXT,
    checkpoint_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (pipeline_id) REFERENCES pipeline(id) ON DELETE CASCADE
);

-- Step 8: Copy data from old table to new table
INSERT INTO pipeline_phase_new (
    id, pipeline_id, phase_path, phase_number, checkpoint_id,
    checkpoint_path, status, created_at, completed_at
)
SELECT
    id, pipeline_id, phase_path, phase_number, checkpoint_id,
    checkpoint_path, status, created_at, completed_at
FROM pipeline_phase;

-- Step 9: Drop the old table
DROP TABLE pipeline_phase;

-- Step 10: Rename the new table
ALTER TABLE pipeline_phase_new RENAME TO pipeline_phase;

-- Step 11: Recreate indexes
CREATE INDEX IF NOT EXISTS idx_pipeline_phase_pipeline ON pipeline_phase(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_phase_path ON pipeline_phase(phase_path);
CREATE INDEX IF NOT EXISTS idx_pipeline_phase_number ON pipeline_phase(phase_number);

-- Verification query (commented out for migration file):
-- SELECT id, pipeline_id, phase_number, phase_path, previous_phase_id FROM pipeline_phase ORDER BY phase_number;
