-- Migration: Add description column to trained_model table
-- Description: Adds a description field to store textual details about how the dataset was combined for training

ALTER TABLE trained_model ADD COLUMN description TEXT;
