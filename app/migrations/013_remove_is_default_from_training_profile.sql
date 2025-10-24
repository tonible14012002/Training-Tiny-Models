-- Migration: Remove is_default from training_argument_profile table
-- Description: Remove default flag logic as default config is hardcoded in TrainerService
-- Created: 2025-10-24

-- Drop the is_default index first
DROP INDEX IF EXISTS idx_training_argument_profile_default;

-- SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
-- Create new table without is_default column
CREATE TABLE IF NOT EXISTS training_argument_profile_new (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    training_config TEXT NOT NULL,
    lora_config TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Copy data from old table to new table
INSERT INTO training_argument_profile_new (id, name, description, training_config, lora_config, created_at, updated_at)
SELECT id, name, description, training_config, lora_config, created_at, updated_at
FROM training_argument_profile;

-- Drop old table
DROP TABLE training_argument_profile;

-- Rename new table to original name
ALTER TABLE training_argument_profile_new RENAME TO training_argument_profile;

-- Recreate index for name lookups
CREATE INDEX IF NOT EXISTS idx_training_argument_profile_name ON training_argument_profile(name);
