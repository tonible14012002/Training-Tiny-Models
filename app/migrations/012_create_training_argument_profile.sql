-- Migration: Create training_argument_profile table
-- Description: Stores reusable training configuration profiles for model training
-- Created: 2025-10-24

-- Training argument profile table - stores training and LoRA configuration profiles
CREATE TABLE IF NOT EXISTS training_argument_profile (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,

    -- Training configuration (JSON string)
    training_config TEXT NOT NULL, -- JSON: learning_rate, batch_size, epochs, etc.

    -- LoRA configuration (JSON string)
    lora_config TEXT NOT NULL, -- JSON: r, lora_alpha, target_modules, etc.

    -- Metadata
    is_default BOOLEAN DEFAULT 0, -- Flag for default profile
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create index for name lookups
CREATE INDEX IF NOT EXISTS idx_training_argument_profile_name ON training_argument_profile(name);

-- Create index for default profile
CREATE INDEX IF NOT EXISTS idx_training_argument_profile_default ON training_argument_profile(is_default);
