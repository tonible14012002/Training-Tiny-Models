-- Migration: Add file_path to composal_dataset table
-- Created: 2025-10-08
-- SQLite Compatible

-- Add file_path column to store the combined/merged dataset file path
ALTER TABLE composal_dataset ADD COLUMN file_path TEXT;
-- This will store the path to the combined dataset file that merges all dataset files
