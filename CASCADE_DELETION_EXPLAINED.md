# Cascade Deletion: What Happens When You Delete a Trained Model

## Short Answer

**YES**, deleting a trained model **automatically deletes all associated evaluation results**.

This is enforced at the database level via the `ON DELETE CASCADE` foreign key constraint.

## How It Works

### Database Schema

The `evaluation_result` table has a foreign key constraint on `trained_model_id`:

```sql
CREATE TABLE evaluation_result (
    id TEXT PRIMARY KEY,
    trained_model_id TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    label_metrics TEXT,
    metrics TEXT,
    evaluated_at TIMESTAMP,
    ...
    FOREIGN KEY (trained_model_id) REFERENCES trained_model(id) ON DELETE CASCADE
    --                                                             ^^^^^^^^^^^^^^^^^
    --                                                             This is the key!
);
```

### What Gets Deleted

When you delete a trained model, the following happens **automatically**:

1. **Trained Model Record** (explicitly deleted by you)
   - `trained_model` table record

2. **Evaluation Results** (CASCADE deleted by database)
   - All `evaluation_result` records where `trained_model_id` matches
   - This includes:
     - Accuracy scores
     - Precision, recall, F1 scores
     - Per-label metrics
     - Additional metrics JSON
     - Evaluation timestamps
     - All other evaluation data

### Deletion Order

SQLite handles the cascade deletion in the correct order:

```
1. You call: DELETE FROM trained_model WHERE id='abc123'
2. SQLite finds: All evaluation_result records with trained_model_id='abc123'
3. SQLite deletes: All those evaluation_result records FIRST
4. SQLite deletes: The trained_model record
5. Transaction commits: All deletions are permanent
```

## Example Scenario

### Before Deletion

```
Database State:
┌─────────────────────────────────────┐
│ trained_model                       │
├─────────────────────────────────────┤
│ id: abc123                          │
│ model_name: Payment Model v0        │
│ model_save_path: .checkpoints/...   │
└─────────────────────────────────────┘
         ▲
         │
         │ (trained_model_id = abc123)
         │
┌─────────────────────────────────────┐
│ evaluation_result                   │
├─────────────────────────────────────┤
│ id: eval1                           │
│ trained_model_id: abc123            │
│ accuracy: 0.95                      │
├─────────────────────────────────────┤
│ id: eval2                           │
│ trained_model_id: abc123            │
│ accuracy: 0.92                      │
├─────────────────────────────────────┤
│ id: eval3                           │
│ trained_model_id: abc123            │
│ accuracy: 0.97                      │
└─────────────────────────────────────┘
```

### After Deletion

```sql
DELETE FROM trained_model WHERE id='abc123';
```

```
Database State:
┌─────────────────────────────────────┐
│ trained_model                       │
├─────────────────────────────────────┤
│ (empty - record deleted)            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ evaluation_result                   │
├─────────────────────────────────────┤
│ (empty - all 3 records deleted)     │
└─────────────────────────────────────┘

Result: All 4 records deleted automatically!
```

## Testing Cascade Deletion

### Using the Test Script

```bash
python test_cascade_deletion.py
```

This script will:
1. Find a trained model with evaluation results
2. Show how many evaluation results exist
3. Explain what will happen with cascade deletion
4. (Optionally) Actually delete and verify cascade works

### Sample Output

```
CASCADE DELETION TEST
================================================================================

1. Finding a trained model with evaluation results...
   ✓ Found model: 93e2e04f-2622-49a1-abdc-7ad26bfe2c2e
   ✓ Model name: Payment Classification Model v0
   ✓ Evaluation results: 3

2. Evaluation results before deletion:
   [0] ID: 0a9e629c-ebb5-4bbb-aedc-d941c7e4608d
       Accuracy: 0.3824884792626728
       F1 Score: 0.33624431268852417
   [1] ID: fca31fcc-c866-45f9-8fa4-ff13284bb45b
       Accuracy: 0.3824884792626728
       F1 Score: 0.33624431268852417
   [2] ID: 8441eed3-8ddf-40a5-8718-40e4baf85367
       Accuracy: 0.3824884792626728
       F1 Score: 0.33624431268852417

3. Testing CASCADE deletion...
   This will delete model: 93e2e04f-2622-49a1-abdc-7ad26bfe2c2e
   Expected: 3 evaluation results will also be deleted
```

## Important Implications

### ⚠️ Warning: Permanent Data Loss

When you delete a trained model:
- **All evaluation metrics are lost** (accuracy, precision, recall, F1)
- **All per-label metrics are lost**
- **All historical evaluation data is lost**
- **This cannot be undone**

### 💡 Best Practices

#### 1. Export Evaluation Data Before Deletion
```python
# Get evaluation results
eval_results = await eval_repo.get_by_trained_model(model_id)

# Export to JSON
import json
with open(f'backup_eval_{model_id}.json', 'w') as f:
    json.dump([{
        'id': e.id,
        'accuracy': e.accuracy,
        'precision': e.precision,
        'recall': e.recall,
        'f1_score': e.f1_score,
        'metrics': e.metrics,
        'evaluated_at': str(e.evaluated_at)
    } for e in eval_results], f, indent=2)

# Then delete
await trained_model_repo.delete(model_id)
```

#### 2. Consider Soft Deletion Instead
Instead of permanently deleting, add a `deleted_at` field:

```sql
-- Add to trained_model table
ALTER TABLE trained_model ADD COLUMN deleted_at TIMESTAMP;

-- Mark as deleted instead of deleting
UPDATE trained_model SET deleted_at = CURRENT_TIMESTAMP WHERE id='abc123';

-- Query only non-deleted models
SELECT * FROM trained_model WHERE deleted_at IS NULL;
```

#### 3. Backup Important Models
Before deleting production models:
```bash
# Backup database
sqlite3 pipeline.db ".backup backup_before_deletion.db"

# Delete model
curl -X POST http://localhost:8000/v2/workflow/delete-model \
  -d '{"model_id": "abc123"}'

# If needed, restore from backup
mv backup_before_deletion.db pipeline.db
```

## What Is NOT Deleted

The following are **NOT** deleted when you delete a trained model:

1. **Dataset files** - These are independent and may be used by other models
2. **Pipeline records** - The pipeline that created the model
3. **Phase records** - The training phase information
4. **Human test sets** - Custom test datasets used for evaluation
5. **Model files** - Only deleted if `delete_files=true` in the API request

### Why These Aren't Deleted

```sql
-- Dataset files: SET NULL on delete
FOREIGN KEY (dataset_file_id) REFERENCES dataset_file(id) ON DELETE SET NULL

-- Phase: CASCADE on delete (but you're not deleting the phase)
FOREIGN KEY (phase_id) REFERENCES pipeline_phase(id) ON DELETE CASCADE

-- Human test set: SET NULL on delete
FOREIGN KEY (human_test_set_id) REFERENCES human_test_set(id) ON DELETE SET NULL
```

## API Behavior

### Delete Model Endpoint

```bash
POST /v2/workflow/delete-model
{
  "model_id": "abc123",
  "delete_files": true
}
```

**What happens:**
1. ✓ Trained model record deleted from database
2. ✓ All evaluation results CASCADE deleted automatically
3. ✓ Model files deleted from disk (if `delete_files=true`)

**Response:**
```json
{
  "message": "Model deleted successfully",
  "model_id": "abc123",
  "model_path": ".checkpoints/pipeline/phase",
  "files_deleted": true,
  "database_deleted": true
}
```

## Verification

### Check Evaluation Results Before Deletion

```sql
-- Find how many evaluation results will be deleted
SELECT COUNT(*) FROM evaluation_result WHERE trained_model_id='abc123';
```

### Verify Cascade Deletion Worked

```sql
-- After deletion, this should return 0
SELECT COUNT(*) FROM evaluation_result WHERE trained_model_id='abc123';
```

### Using Python

```python
from app.core.repositories import evaluation_result_repository as eval_repo

# Before deletion
eval_results_before = await eval_repo.get_by_trained_model(model_id)
print(f"Evaluation results before: {len(eval_results_before)}")

# Delete model
await trained_model_repo.delete(model_id)
await session.commit()

# After deletion
eval_results_after = await eval_repo.get_by_trained_model(model_id)
print(f"Evaluation results after: {len(eval_results_after)}")  # Should be 0
```

## Summary

✅ **YES** - Evaluation results are automatically deleted when you delete a trained model

✅ **Automatic** - Happens at database level via `ON DELETE CASCADE`

✅ **Atomic** - All deletions happen in one transaction

⚠️ **Permanent** - Cannot be undone

💾 **Backup** - Export evaluation data before deletion if needed

📝 **Logged** - All deletions are logged in application logs

## Related Documentation

- See `DELETE_MODEL_ENDPOINT.md` for full API documentation
- See `test_cascade_deletion.py` for verification testing
- See migration `002_add_training_evaluation_tables.sql` for schema details
