# Delete Model Endpoint Fix

## Problem

The delete model endpoint was deleting the trained model from the database and removing the physical files, but the **phase detail API was still showing the model information** in the phase's checkpoint fields.

## Root Cause

The `pipeline_phase` table has two fields that reference the trained model:
- `checkpoint_id` - ID of the checkpoint
- `checkpoint_path` - Path to the model files

When deleting a trained model, the endpoint was:
1. ✅ Deleting the record from `trained_model` table
2. ✅ Deleting the physical files from disk
3. ✅ CASCADE deleting evaluation results
4. ❌ **NOT clearing the phase's checkpoint reference**

This caused the phase to still reference a deleted model, leading to:
- Phase detail API showing invalid checkpoint information
- Inconsistent data between `pipeline_phase` and `trained_model` tables
- Confusion about which models are actually available

## Solution

Updated the delete endpoint to **clear the phase's checkpoint reference** if the deleted model is the phase's checkpoint.

### Code Changes

**File:** `app/api/routes/v2/workflow.py:2070-2085`

**Added logic:**
```python
# Check if this model is referenced in any phase's checkpoint
# If so, we should clear the phase's checkpoint_path and checkpoint_id
phase_repo = repos.PhaseRepository(db)
phase = await phase_repo.get_by_id(trained_model.phase_id)

if phase and phase.checkpoint_path == model_path:
    # This model is the phase's checkpoint - clear it
    logger.info(f"Clearing phase checkpoint reference for phase: {phase.id}")
    await phase_repo.update(
        phase.id,
        checkpoint_id=None,
        checkpoint_path=None
    )
    await db.commit()
    phase_checkpoint_cleared = True
```

### What This Does

1. **Retrieves the phase** that owns the trained model
2. **Checks if the model is the phase's checkpoint** by comparing paths
3. **Clears the checkpoint reference** if they match:
   - Sets `checkpoint_id` to `NULL`
   - Sets `checkpoint_path` to `NULL`
4. **Commits the change** to the database
5. **Reports the action** in the response

## Updated Response

### New Field

Added `phase_checkpoint_cleared` to the response:

```json
{
  "message": "Model deleted successfully",
  "model_id": "abc123-model-id",
  "model_path": ".checkpoints/pipeline_id/phase_number",
  "files_deleted": true,
  "database_deleted": true,
  "phase_checkpoint_cleared": true  // ← NEW FIELD
}
```

### Response Schema Update

**File:** `app/core/schemas/workflow.py:602-609`

```python
class DeleteModelResponse(BaseModel):
    message: str
    model_id: str
    model_path: Optional[str] = None
    files_deleted: bool
    database_deleted: bool
    phase_checkpoint_cleared: bool  # ← NEW FIELD
```

## Deletion Flow (Updated)

### Before (Broken)
```
1. Delete files from disk ✅
2. Delete trained_model record from database ✅
3. CASCADE delete evaluation results ✅
4. ❌ Phase still has checkpoint_path pointing to deleted model
```

### After (Fixed)
```
1. Delete files from disk ✅
2. Get phase and check if this model is its checkpoint ✅
3. Clear phase checkpoint reference if match ✅
4. Delete trained_model record from database ✅
5. CASCADE delete evaluation results ✅
```

## Testing

### Before Fix
```bash
# Delete a model
POST /v2/workflow/delete-model
{"model_id": "abc123"}

# Response
{
  "database_deleted": true,
  "files_deleted": true
}

# Phase detail still shows the model
GET /v2/workflow/phase-detail?phase_id=xyz
{
  "checkpoint_path": ".checkpoints/pipeline/phase",  // ❌ Still present
  "checkpoint_id": "checkpoint_v1"                    // ❌ Still present
}
```

### After Fix
```bash
# Delete a model
POST /v2/workflow/delete-model
{"model_id": "abc123"}

# Response
{
  "database_deleted": true,
  "files_deleted": true,
  "phase_checkpoint_cleared": true  // ✅ Cleared
}

# Phase detail no longer shows the model
GET /v2/workflow/phase-detail?phase_id=xyz
{
  "checkpoint_path": null,  // ✅ Cleared
  "checkpoint_id": null     // ✅ Cleared
}
```

## Impact

### Positive Changes
✅ Phase detail API now shows accurate information
✅ Data consistency between phase and trained_model tables
✅ No orphaned checkpoint references
✅ Clear indication in response when checkpoint is cleared

### Breaking Changes
⚠️ Response schema changed (added new field)
- Old clients will still work (new field is additive)
- New clients can use `phase_checkpoint_cleared` for better UX

## Edge Cases Handled

### Case 1: Model is NOT the phase checkpoint
```python
# Phase has different checkpoint
phase.checkpoint_path = ".checkpoints/other/model"
trained_model.model_save_path = ".checkpoints/target/model"

# Result: phase_checkpoint_cleared = False
# Phase checkpoint is not modified
```

### Case 2: Model IS the phase checkpoint
```python
# Phase has this model as checkpoint
phase.checkpoint_path = ".checkpoints/target/model"
trained_model.model_save_path = ".checkpoints/target/model"

# Result: phase_checkpoint_cleared = True
# Phase checkpoint is cleared (set to NULL)
```

### Case 3: Phase has no checkpoint
```python
# Phase has no checkpoint set
phase.checkpoint_path = None
phase.checkpoint_id = None

# Result: phase_checkpoint_cleared = False
# Nothing to clear
```

### Case 4: Model files don't exist
```python
# Files already deleted manually
model_path = ".checkpoints/target/model"
os.path.exists(model_path) = False

# Result:
# - files_deleted = False
# - database_deleted = True
# - phase_checkpoint_cleared = True (if it was the checkpoint)
# Database cleanup still happens
```

## Files Modified

1. **app/api/routes/v2/workflow.py:2036-2105**
   - Added `phase_checkpoint_cleared` variable
   - Added phase checkpoint clearing logic
   - Updated response to include new field

2. **app/core/schemas/workflow.py:602-609**
   - Added `phase_checkpoint_cleared` field to `DeleteModelResponse`

3. **DELETE_MODEL_ENDPOINT.md**
   - Updated response schema documentation
   - Added explanation of phase checkpoint clearing
   - Updated deletion logic documentation

## Related Tables

### pipeline_phase
```sql
CREATE TABLE pipeline_phase (
    id TEXT PRIMARY KEY,
    checkpoint_id TEXT,      -- ← Cleared when model deleted
    checkpoint_path TEXT,    -- ← Cleared when model deleted
    ...
);
```

### trained_model
```sql
CREATE TABLE trained_model (
    id TEXT PRIMARY KEY,
    phase_id TEXT,           -- Links to pipeline_phase
    model_save_path TEXT,    -- Compared with checkpoint_path
    ...
);
```

## Summary

✅ **Fixed:** Phase checkpoint reference now cleared when deleting models
✅ **Added:** `phase_checkpoint_cleared` field in response
✅ **Improved:** Data consistency between tables
✅ **Enhanced:** Better user experience with accurate phase information

The delete endpoint now properly handles all aspects of model deletion, ensuring no orphaned references remain in the database.
