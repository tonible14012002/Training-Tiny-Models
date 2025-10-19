# Endpoint Audit Results

## Summary

Audited all endpoints in `app/api/routes/v2/workflow.py` for correctness after the phase relationship refactoring. Found and fixed **3 critical issues** related to composal dataset retrieval.

## Issues Found and Fixed

### ✅ Issue 1: `evaluate_human_set` endpoint (Line 582)

**Problem**: Used `get_by_pipeline()` which returns ANY composal dataset from the pipeline, not the correct one for this phase's sequence.

**Before:**
```python
base_ds = (await repos.DatasetRepository(db).get_by_pipeline(pipeline.id))[0]
```

**After:**
```python
# Get the parent (root) phase's composal dataset for this sequence
parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
if not parent_phase:
    # This IS the parent phase
    parent_phase = phase

base_ds = (await repos.DatasetRepository(db).get_by_phase(parent_phase.id))[0]
```

**Why it matters**: In a pipeline with multiple sequences (multiple root phases), the old code could retrieve the wrong composal dataset, leading to incorrect training data.

---

### ✅ Issue 2: `generate_error_bucket_samples` endpoint (Line 818)

**Problem**: Same issue - used `get_by_pipeline()` instead of getting the parent phase's composal dataset.

**Before:**
```python
base_ds = (await repos.DatasetRepository(db).get_by_pipeline(phase.pipeline_id))[0]
```

**After:**
```python
# Get the parent (root) phase's composal dataset for this sequence
parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
if not parent_phase:
    # This IS the parent phase
    parent_phase = phase

base_ds = (await repos.DatasetRepository(db).get_by_phase(parent_phase.id))[0]
```

**Why it matters**: Error bucket sample generation needs to use the correct composal dataset from the phase's sequence to analyze low-confidence samples properly.

---

### ✅ Issue 3: `get_training_pool` endpoint (Line 1211-1212)

**Problem**: Used `get_by_phase(phase_id)` which only returns composal datasets created by this specific phase. For child phases, this returns empty since they don't have their own composal dataset - they share the parent's.

**Before:**
```python
# Get composal datasets for this phase
composal_datasets = await repos.DatasetRepository(db).get_by_phase(phase_id)
if not composal_datasets:
    return {
        "error": "No training pool found",
        "message": "No composal dataset exists for this phase. Run first-gen to create one."
    }
```

**After:**
```python
# Get the parent (root) phase's composal dataset for this sequence
# All phases in a sequence share the same composal dataset from the parent
parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
if not parent_phase:
    # This IS the parent phase
    parent_phase = phase

# Get composal datasets for the parent phase
composal_datasets = await repos.DatasetRepository(db).get_by_phase(parent_phase.id)
if not composal_datasets:
    return {
        "error": "No training pool found",
        "message": "No composal dataset exists for this phase sequence. Run first-gen to create one."
    }
```

**Why it matters**: Child phases would fail to load training pool data since they don't have their own composal dataset. The fix ensures all phases in a sequence can access the shared composal dataset.

---

## Endpoints Verified as Correct

### ✅ `first_generation` endpoint (Line 124-239)

**Status**: Correct

**Details**:
- Properly creates base phase with `previous_phase_id=None`
- Creates composal dataset linked to the base phase
- Correct phase path: `phase_path=""`

---

### ✅ `continue_generation` endpoint (Line 861-1015)

**Status**: Correct

**Details**:
- Validates that provided phase_id is a root phase (`phase_path=""`)
- Uses `get_child_phases()` to find all descendants
- Correctly identifies the latest/previous phase
- Creates new phase as child of previous phase using `previous_phase_id`
- Links dataset file to parent phase's composal dataset

---

### ✅ `view_phase_detail` endpoint (Line 1271-1335)

**Status**: Correct

**Details**:
- Only shows child_phases for base phases (`if not phase.phase_path`)
- Correctly uses `get_child_phases()` to retrieve all descendants
- Proper phase hierarchy representation

---

## Key Learnings

### 1. Composal Dataset Sharing

All phases in a sequence share the same composal dataset from the parent (root) phase. When retrieving composal datasets:

- **For parent phases**: Use `get_by_phase(phase.id)` directly
- **For child phases**: Must first get parent phase, then use `get_by_phase(parent_phase.id)`

### 2. Pattern for Getting Parent Phase's Composal Dataset

```python
# Get the parent (root) phase's composal dataset
parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
if not parent_phase:
    # This IS the parent phase
    parent_phase = phase

base_ds = (await repos.DatasetRepository(db).get_by_phase(parent_phase.id))[0]
```

### 3. Never Use `get_by_pipeline()` for Composal Datasets

Using `get_by_pipeline()` is unsafe because:
- A pipeline can have multiple sequences (multiple root phases)
- It returns the first composal dataset it finds, which might be from the wrong sequence
- Always use `get_by_phase(parent_phase.id)` instead

### 4. Phase Relationship Semantics

- **parent_phase**: Root phase with `phase_path=""` (base of sequence)
- **previous_phase**: Immediate predecessor (last phase before current)
- **child_phases**: All descendants in the sequence

## Testing Recommendations

After these fixes, test the following scenarios:

1. **Single Sequence**:
   - Create base phase → train → evaluate → continue-gen
   - Verify all endpoints can access composal dataset

2. **Multiple Sequences**:
   - Create two separate base phases in same pipeline
   - Verify each sequence maintains its own composal dataset
   - Ensure no cross-contamination between sequences

3. **Deep Hierarchies**:
   - Create chain: Phase 0 → Phase 1 → Phase 2 → Phase 3
   - Call evaluate/error-bucket-gen on Phase 3
   - Verify it uses Phase 0's composal dataset

4. **Training Pool Access**:
   - Call `/phase/{phase_id}/trainingpool` on child phases
   - Verify it returns parent's composal dataset data

## Conclusion

All endpoints now correctly:
1. Use `get_parent_phase()` to find the root phase
2. Retrieve composal datasets from the parent phase
3. Maintain proper phase hierarchy semantics
4. Support multiple sequences in the same pipeline

The refactoring to use `phase_path` for all relationships is now fully consistent across all endpoints.
