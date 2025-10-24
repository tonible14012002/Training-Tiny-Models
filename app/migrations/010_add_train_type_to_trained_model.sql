-- Migration: Add train_type column to trained_model table
-- Created: 2025-10-20
-- SQLite Compatible
--
-- This migration adds a train_type column to support different training types.
-- The train_type field is an extendable enum (string) that can have values like:
-- 'FROM_SCRATCH', 'CONTINUE', etc.

-- Add train_type column to trained_model table
ALTER TABLE trained_model ADD COLUMN train_type TEXT;

-- Create index for efficient train_type filtering
CREATE INDEX IF NOT EXISTS idx_trained_model_train_type ON trained_model(train_type);

-- Create composite index for phase_id + train_type to support uniqueness checks
CREATE INDEX IF NOT EXISTS idx_trained_model_phase_train_type ON trained_model(phase_id, train_type);

-- Verification query (commented out for migration file):
-- SELECT id, phase_id, model_name, train_type FROM trained_model;
