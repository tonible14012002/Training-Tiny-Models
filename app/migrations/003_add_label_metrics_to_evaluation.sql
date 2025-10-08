-- Migration: Add label_metrics field to evaluation_result table
-- Created: 2025-10-07
-- SQLite Compatible

-- Add label_metrics column to store per-label detailed metrics
ALTER TABLE evaluation_result ADD COLUMN label_metrics TEXT;
-- JSON format example: {"payment_request": {"precision": 0.95, "recall": 0.92, "f1_score": 0.93, "support": 100}, "payment_intent": {...}}
