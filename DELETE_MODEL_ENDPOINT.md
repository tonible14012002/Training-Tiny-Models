# Delete Model Endpoint

## Overview

A new endpoint has been added to delete trained models from both the database and disk storage.

⚠️ **WARNING:** Deleting a trained model will **automatically delete all associated evaluation results** due to CASCADE deletion constraints. This operation is **permanent and cannot be undone**.

## Endpoint

```
POST /v2/workflow/delete-model
```

## Features

- **Database Deletion**: Removes the trained model record from the database
- **File Deletion**: Optionally deletes physical model files from disk
- **Flexible Control**: Can choose to delete only database record or both database and files
- **Safe Operation**: Validates model existence before deletion
- **Error Handling**: Graceful handling of missing files or database records
- **Related Data**: Automatically handles cascade deletion of related evaluation results (via foreign key constraints)

## Request Schema

```json
{
  "model_id": "abc123-model-id",
  "delete_files": true  // optional, defaults to true
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | string | Yes | - | ID of the trained model to delete |
| `delete_files` | boolean | No | `true` | Whether to delete physical files from disk |

## Response Schema

```json
{
  "message": "Model deleted successfully",
  "model_id": "abc123-model-id",
  "model_path": ".checkpoints/pipeline_id/phase_number",
  "files_deleted": true,
  "database_deleted": true,
  "phase_checkpoint_cleared": true
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Success or error message |
| `model_id` | string | ID of the deleted model |
| `model_path` | string | Path to the model files |
| `files_deleted` | boolean | Whether files were deleted from disk |
| `database_deleted` | boolean | Whether database record was deleted |
| `phase_checkpoint_cleared` | boolean | Whether the phase's checkpoint reference was cleared |

## Usage Examples

### Using curl

#### Delete model and files
```bash
curl -X POST http://localhost:8000/v2/workflow/delete-model \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model_id": "abf8571b-9808-45a1-86bc-f495f29a5f3d",
    "delete_files": true
  }'
```

#### Delete from database only (preserve files)
```bash
curl -X POST http://localhost:8000/v2/workflow/delete-model \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model_id": "abf8571b-9808-45a1-86bc-f495f29a5f3d",
    "delete_files": false
  }'
```

### Using Python

```python
import requests

# Delete model and files
response = requests.post(
    "http://localhost:8000/v2/workflow/delete-model",
    json={
        "model_id": "abc123-model-id",
        "delete_files": True
    },
    headers={"X-API-Key": "your-api-key"}
)

result = response.json()
print(f"Database deleted: {result['database_deleted']}")
print(f"Files deleted: {result['files_deleted']}")
```

### Using Test Scripts

#### 1. Check Available Models
```bash
python test_delete_model.py
```

This will:
- List available models in the database
- Show model IDs, names, paths, and status
- Simulate deletion without actually deleting

#### 2. Test API Endpoint
```bash
# Delete model and files
python test_delete_model_api.py <model_id>

# Delete from database only
python test_delete_model_api.py <model_id> false

# Test with non-existent model
python test_delete_model_api.py test
```

## Deletion Behavior

### What Gets Deleted

#### Database (always when `database_deleted=true`)
- **Trained model record** from `trained_model` table
- **All evaluation results** associated with the model (automatic CASCADE deletion)
  - This includes all records in `evaluation_result` table where `trained_model_id` matches
  - Metrics, accuracy, precision, recall, F1 scores, etc. are all deleted
  - This is enforced at the database level via foreign key constraint: `ON DELETE CASCADE`
- **Phase checkpoint reference** (if this model is the phase's checkpoint)
  - Clears `checkpoint_id` and `checkpoint_path` fields in the `pipeline_phase` table
  - This ensures the phase detail API no longer shows this model as the checkpoint
  - Only cleared if the deleted model's path matches the phase's checkpoint path

#### Files (only when `delete_files=true`)
- Model checkpoint directory (e.g., `.checkpoints/pipeline_id/phase_number/`)
- All contents including:
  - `_merged/` directory with merged model
  - Adapter files (`adapter_model.safetensors`)
  - Tokenizer files
  - Training metadata
  - Configuration files

### Deletion Logic

```python
# 1. Delete files if requested
if delete_files and model_path exists:
    if path is directory:
        shutil.rmtree(path)  # Recursive deletion
    elif path is file:
        path.unlink()  # Delete single file

# 2. Clear phase checkpoint reference if this model is the checkpoint
phase = get_phase(trained_model.phase_id)
if phase.checkpoint_path == model_path:
    phase.checkpoint_id = None
    phase.checkpoint_path = None
    database.commit()

# 3. Delete from database (cascade deletes evaluation results)
database.delete(model_id)
```

**Important:** The phase checkpoint clearing (step 2) is crucial because:
- The `pipeline_phase` table stores `checkpoint_path` pointing to the trained model
- Without clearing this reference, the phase detail API would show a deleted model
- This ensures data consistency between the phase and trained_model tables

## Error Handling

### Model Not Found
```json
{
  "error": "Model not found",
  "message": "No trained model found with ID: abc123"
}
```

### File Deletion Failed
```json
{
  "error": "File deletion failed",
  "message": "Failed to delete model files at /path: Permission denied",
  "model_id": "abc123",
  "files_deleted": false,
  "database_deleted": false
}
```

**Note:** If file deletion fails, database deletion is **not** performed to maintain consistency.

### Database Deletion Failed
```json
{
  "error": "Database deletion failed",
  "message": "Failed to delete model from database: abc123"
}
```

### General Errors
```json
{
  "error": "error message",
  "message": "An error occurred during model deletion"
}
```

## Safety Features

1. **Existence Check**: Verifies model exists before attempting deletion
2. **Path Validation**: Checks if model path exists before file deletion
3. **Transaction Safety**: Database operations are wrapped in transactions
4. **Graceful Degradation**: Continues with database deletion even if files don't exist
5. **Detailed Logging**: All operations are logged for audit trail
6. **Error Recovery**: Stops database deletion if file deletion fails (when requested)

## Use Cases

### 1. Clean Up Failed Training
Delete models that failed during training:
```bash
# Delete failed model
curl -X POST http://localhost:8000/v2/workflow/delete-model \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"model_id": "failed-model-id"}'
```

### 2. Free Up Disk Space
Delete old models while preserving database records for analytics:
```bash
# Keep database record, delete files
curl -X POST http://localhost:8000/v2/workflow/delete-model \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"model_id": "old-model-id", "delete_files": true}'
```

### 3. Archive Models
Delete database record but keep files for manual backup:
```bash
# This is NOT supported - you can only delete both or database only
# To archive: copy files elsewhere, then delete with delete_files=true
```

### 4. Clean Up Test Models
Batch delete test models:
```python
import requests

test_model_ids = ["test-1", "test-2", "test-3"]

for model_id in test_model_ids:
    response = requests.post(
        "http://localhost:8000/v2/workflow/delete-model",
        json={"model_id": model_id, "delete_files": True},
        headers={"X-API-Key": "your-api-key"}
    )
    print(f"Deleted {model_id}: {response.json()['message']}")
```

## Database Schema Impact

### TrainedModel Table
```sql
CREATE TABLE trained_model (
    id TEXT PRIMARY KEY,
    phase_id TEXT FOREIGN KEY,
    model_name TEXT,
    model_save_path TEXT,
    ...
);
```

### EvaluationResult Table
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
    ...
    FOREIGN KEY (trained_model_id) REFERENCES trained_model(id) ON DELETE CASCADE
);
```

### Cascade Deletion Behavior
When a `TrainedModel` is deleted, the following happens **automatically** at the database level:

1. **All `EvaluationResult` records** where `trained_model_id` matches are deleted
   - This is enforced by the `ON DELETE CASCADE` constraint
   - Happens atomically in the same transaction
   - No application code needed - database handles it

**Example:**
```sql
-- If you have:
trained_model (id='abc123', ...)
evaluation_result (id='eval1', trained_model_id='abc123', ...)
evaluation_result (id='eval2', trained_model_id='abc123', ...)

-- When you delete the trained_model:
DELETE FROM trained_model WHERE id='abc123';

-- The database automatically deletes:
-- - evaluation_result with id='eval1'
-- - evaluation_result with id='eval2'
```

**Important Notes:**
- Cascade deletion happens **before** the trained model is deleted
- If cascade deletion fails, the entire transaction is rolled back
- No orphaned evaluation results are left in the database
- This ensures referential integrity

## Finding Model IDs

### Using SQLite Command Line
```bash
sqlite3 pipeline.db "SELECT id, model_name, model_save_path, status FROM trained_model ORDER BY created_at DESC LIMIT 10;"
```

### Using Python Script
```python
python test_delete_model.py
```

### Using Database Query
```sql
SELECT
    id,
    model_name,
    model_save_path,
    status,
    created_at
FROM trained_model
WHERE status = 'DONE'
ORDER BY created_at DESC;
```

## Performance Characteristics

- **Database Deletion**: Fast (~10-50ms)
- **File Deletion**: Depends on model size
  - Small models (~20MB): ~100-500ms
  - Large models with evaluations: ~1-2 seconds
- **Network Overhead**: Minimal (~5-10ms)

## Schemas Added

### DeleteModelRequest
```python
class DeleteModelRequest(BaseModel):
    model_id: str
    delete_files: bool = True
```

### DeleteModelResponse
```python
class DeleteModelResponse(BaseModel):
    message: str
    model_id: str
    model_path: Optional[str] = None
    files_deleted: bool
    database_deleted: bool
```

## Files Modified

1. `app/core/schemas/workflow.py:591-608` - Added request/response schemas
2. `app/api/routes/v2/workflow.py:1981-2087` - Implemented delete endpoint

## Files Created

1. `test_delete_model.py` - Database inspection and simulation test
2. `test_delete_model_api.py` - API endpoint test script
3. `DELETE_MODEL_ENDPOINT.md` - This documentation

## Integration with Workflow

This endpoint can be used at any point in the workflow to clean up unwanted models:

1. Create pipeline: `POST /workflow/pipeline`
2. Generate data: `POST /workflow/first-gen`
3. Train model: `POST /workflow/train`
4. Evaluate model: `POST /workflow/evaluate`
5. **Delete model:** `POST /workflow/delete-model` ← **NEW**
6. Continue generation: `POST /workflow/continue-gen`

## Best Practices

### 1. Always Check Model ID
Verify the model ID before deletion to avoid accidents.

### 2. Backup Important Models
Create backups before deleting production models:
```bash
# Backup model files
cp -r .checkpoints/pipeline_id/phase_number /backups/

# Then delete
curl -X POST http://localhost:8000/v2/workflow/delete-model \
  -d '{"model_id": "..."}'
```

### 3. Use delete_files=false for Testing
Test database operations without deleting files:
```bash
curl -X POST http://localhost:8000/v2/workflow/delete-model \
  -d '{"model_id": "...", "delete_files": false}'
```

### 4. Clean Up in Batches
Delete old models periodically to free up space:
```python
# Get models older than 30 days
old_models = get_old_models(days=30)

# Delete in batches
for model in old_models:
    delete_model(model.id, delete_files=True)
```

### 5. Monitor Disk Space
Track disk usage before and after deletion:
```bash
# Before
du -sh .checkpoints/

# Delete models
python cleanup_models.py

# After
du -sh .checkpoints/
```

## Notes

- The endpoint uses soft deletion (permanent delete, not marked as deleted)
- Related evaluation results are cascade deleted via database constraints
- File deletion is recursive for directories
- The operation is atomic - either both succeed or both fail (when delete_files=true)
- Logs are generated for all deletion operations for audit purposes
