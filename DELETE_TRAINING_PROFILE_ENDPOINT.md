# Delete Training Profile Endpoint

## Overview

An enhanced endpoint to delete training argument profiles with protection for profiles in use by trained models.

## Endpoint

```
DELETE /v2/workflow/training-profile/{profile_id}?force=false
```

## Features

- **Usage Check**: Verifies if any trained models are using the profile
- **Safe Deletion**: Prevents accidental deletion of profiles in use
- **Force Deletion**: Optionally clears model references before deletion
- **Detailed Response**: Reports how many models were affected
- **Error Protection**: Clear error messages when profile is in use

## Parameters

| Parameter | Type | Location | Required | Default | Description |
|-----------|------|----------|----------|---------|-------------|
| `profile_id` | string | Path | Yes | - | ID of the training profile to delete |
| `force` | boolean | Query | No | `false` | Force deletion even if models are using this profile |

## Response Schema

```json
{
  "message": "Training argument profile deleted successfully",
  "profile_id": "abc123-profile-id",
  "profile_name": "fast-training",
  "database_deleted": true,
  "models_updated": 3
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Success or error message |
| `profile_id` | string | ID of the deleted profile |
| `profile_name` | string | Name of the deleted profile |
| `database_deleted` | boolean | Whether the profile was deleted from database |
| `models_updated` | integer | Number of models that had their profile_id cleared |

## Usage Examples

### 1. Attempt to Delete Profile (Safe Mode)

```bash
curl -X DELETE http://localhost:8000/v2/workflow/training-profile/abc123 \
  -H "X-API-Key: your-api-key"
```

**Response (if profile is in use):**
```json
{
  "error": "Profile in use",
  "message": "Cannot delete profile 'fast-training'. 3 trained model(s) are using this profile. Use force=true to delete anyway (will set model profile_id to NULL).",
  "models_count": 3
}
```

**Response (if profile is NOT in use):**
```json
{
  "message": "Training argument profile deleted successfully",
  "profile_id": "abc123",
  "profile_name": "fast-training",
  "database_deleted": true,
  "models_updated": 0
}
```

### 2. Force Delete Profile

```bash
curl -X DELETE "http://localhost:8000/v2/workflow/training-profile/abc123?force=true" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "message": "Training argument profile deleted successfully",
  "profile_id": "abc123",
  "profile_name": "fast-training",
  "database_deleted": true,
  "models_updated": 3
}
```

### 3. Using Python

```python
import requests

# Safe deletion (will fail if in use)
response = requests.delete(
    "http://localhost:8000/v2/workflow/training-profile/abc123",
    headers={"X-API-Key": "your-api-key"}
)

# Force deletion (clears references)
response = requests.delete(
    "http://localhost:8000/v2/workflow/training-profile/abc123",
    params={"force": True},
    headers={"X-API-Key": "your-api-key"}
)

result = response.json()
print(f"Deleted: {result['database_deleted']}")
print(f"Models updated: {result['models_updated']}")
```

## Deletion Behavior

### Scenario 1: Profile NOT in use (force=false or force=true)

```
1. Check if any models use this profile
   └─ Result: No models found
2. Delete profile from database
   └─ Result: Profile deleted
3. Return success
   └─ models_updated = 0
```

### Scenario 2: Profile IN use + force=false (DEFAULT)

```
1. Check if any models use this profile
   └─ Result: 3 models found
2. Return error
   └─ Error: "Profile in use"
   └─ models_count: 3
3. Profile NOT deleted (protection)
```

### Scenario 3: Profile IN use + force=true

```
1. Check if any models use this profile
   └─ Result: 3 models found
2. Clear profile reference from all models
   ├─ Model 1: training_argument_profile_id = NULL
   ├─ Model 2: training_argument_profile_id = NULL
   └─ Model 3: training_argument_profile_id = NULL
3. Commit changes
4. Delete profile from database
5. Return success
   └─ models_updated = 3
```

## What Gets Deleted/Updated

### Database Changes (force=false, profile not in use)
- ✅ Training profile record from `training_argument_profile` table

### Database Changes (force=true, profile in use)
- ✅ Training profile record from `training_argument_profile` table
- ✅ Clears `training_argument_profile_id` to `NULL` in all trained models using this profile

### What is NOT Deleted
- ❌ Trained models (only their profile reference is cleared)
- ❌ Model files on disk
- ❌ Evaluation results
- ❌ Dataset files

## Error Handling

### Profile Not Found
```json
{
  "error": "Training argument profile not found",
  "message": "No profile found with ID: abc123"
}
```

### Profile In Use (without force)
```json
{
  "error": "Profile in use",
  "message": "Cannot delete profile 'fast-training'. 3 trained model(s) are using this profile. Use force=true to delete anyway (will set model profile_id to NULL).",
  "models_count": 3
}
```

### Deletion Failed
```json
{
  "error": "Deletion failed",
  "message": "Failed to delete profile: abc123"
}
```

### Unexpected Error
```json
{
  "error": "error message",
  "message": "An unexpected error occurred"
}
```

## Database Schema Impact

### training_argument_profile Table
```sql
CREATE TABLE training_argument_profile (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    training_config TEXT NOT NULL,
    lora_config TEXT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### trained_model Table
```sql
CREATE TABLE trained_model (
    id TEXT PRIMARY KEY,
    training_argument_profile_id TEXT,  -- ← Set to NULL when force=true
    ...
);
```

**Note:** There is NO CASCADE deletion constraint between `training_argument_profile` and `trained_model`. The relationship is enforced at the application level.

## Use Cases

### 1. Clean Up Unused Profiles

```bash
# Safe deletion - only works if not in use
DELETE /v2/workflow/training-profile/old-profile-id
```

### 2. Replace Profile with New Version

```bash
# 1. Create new profile with updated settings
POST /v2/workflow/training-profile
{"name": "fast-training-v2", ...}

# 2. Manually update models to use new profile (if needed)
# (via training or re-training)

# 3. Delete old profile
DELETE /v2/workflow/training-profile/old-profile-id
```

### 3. Force Delete Profile Regardless of Usage

```bash
# Delete profile and clear all model references
DELETE /v2/workflow/training-profile/abc123?force=true
```

**Warning:** Models will lose their training configuration reference. They will still have the configuration in `training_params` JSON field, but won't link to a reusable profile.

### 4. Check Profile Usage Before Deletion

```bash
# Attempt deletion without force
DELETE /v2/workflow/training-profile/abc123

# If error with models_count, decide whether to force delete
# Then delete with force if needed
DELETE /v2/workflow/training-profile/abc123?force=true
```

## Safety Features

1. **Default Protection**: By default (force=false), prevents deletion of profiles in use
2. **Usage Count**: Shows exactly how many models are using the profile
3. **Explicit Force**: Requires explicit `force=true` to delete profiles in use
4. **Clear Messaging**: Error messages explain the situation and how to proceed
5. **Model Preservation**: Models are never deleted, only their profile reference is cleared

## Testing

### Test Scenario 1: Delete Unused Profile

```bash
# Create a profile
POST /v2/workflow/training-profile
{
  "name": "test-profile",
  "training_config": {...},
  "lora_config": {...}
}

# Delete immediately (no models using it)
DELETE /v2/workflow/training-profile/{profile_id}

# Expected: Success, models_updated=0
```

### Test Scenario 2: Delete Profile in Use (Protected)

```bash
# Profile is used by 3 models
DELETE /v2/workflow/training-profile/abc123

# Expected: Error with models_count=3
```

### Test Scenario 3: Force Delete Profile in Use

```bash
# Profile is used by 3 models
DELETE /v2/workflow/training-profile/abc123?force=true

# Expected: Success, models_updated=3
# Verify: All 3 models now have training_argument_profile_id=NULL
```

## Comparison with Delete Model Endpoint

| Feature | Delete Model | Delete Training Profile |
|---------|-------------|------------------------|
| Deletes database record | ✅ Yes | ✅ Yes |
| Deletes files | ✅ Yes (if requested) | ❌ No (profiles don't have files) |
| CASCADE deletes related data | ✅ Yes (evaluations) | ❌ No (models preserved) |
| Clears references | ✅ Yes (phase checkpoint) | ✅ Yes (model profile_id if force=true) |
| Force parameter | ❌ No | ✅ Yes |
| Usage check | ❌ No | ✅ Yes |

## Files Modified

1. **app/api/routes/v2/workflow.py:1759-1864**
   - Enhanced delete_training_profile endpoint
   - Added force parameter
   - Added usage check
   - Added model reference clearing

2. **app/core/schemas/workflow.py:611-628**
   - Added DeleteTrainingProfileRequest schema
   - Added DeleteTrainingProfileResponse schema

## Notes

- Training profiles store reusable configuration for training models
- Profiles contain training_config (learning rate, batch size, etc.) and lora_config (LoRA parameters)
- Models can reference a profile via `training_argument_profile_id`
- When force deleting, models retain their configuration in `training_params` JSON field
- The profile name must be unique, so you cannot recreate a deleted profile with the same name immediately

## Best Practices

### 1. Check Before Deleting
Always try deletion without force first to see if the profile is in use.

### 2. Update Models First
If possible, update models to use a different profile before deleting the old one.

### 3. Document Force Deletions
When using force deletion, document which models were affected for future reference.

### 4. Consider Archiving
Instead of deleting, consider renaming the profile with an "archived-" prefix to preserve the configuration.

### 5. Backup Important Profiles
Export profile configurations before deletion:
```bash
GET /v2/workflow/training-profile/{profile_id}
# Save the response for later restoration
```

## Summary

✅ **Enhanced**: Now checks for profile usage before deletion
✅ **Protected**: Prevents accidental deletion of profiles in use
✅ **Flexible**: Force parameter allows deletion when needed
✅ **Informative**: Returns detailed information about affected models
✅ **Safe**: Models are preserved, only references are cleared

The delete training profile endpoint now provides safe and controlled deletion with clear feedback about the impact.
