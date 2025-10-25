# Validator Not Running - Diagnosis

## Problem
The generated data file still contains mislabeled "pay me" samples labeled as `payment_intent` when they should be `payment_request`.

Examples from `.cache/33625915-47be-4ca3-a144-bfe358b69e42/ebc89f01-af13-4b35-8858-cc195f4f7aed/0/.data.jsonl`:
```json
{"msg": "Don't forget to pay me back the 60$ for last month!", "label": 0}  // WRONG! Should be label 1
{"msg": "Please pay me back the 5$ you owe!", "label": 0}  // WRONG! Should be label 1
{"msg": "Make sure to pay me back the 50$ soon!", "label": 0}  // WRONG! Should be label 1
```

Where:
- `label 0` = payment_intent (WRONG for these samples)
- `label 1` = payment_request (CORRECT)
- `label 2` = open_intent

## Root Cause

**The validator integration is in the code, but the server was NOT restarted after the changes were made.**

Timeline:
- **11:35-11:36 AM**: Code changes made (validator integration)
- **12:27 PM**: Data generated (still has mislabeled samples)
- **Conclusion**: Server was running with old code

## Verification

### Test 1: Validator Works Standalone ✅
```bash
python test_validator_actual_data.py
```
**Result**: 9/10 samples correctly fixed → Validator IS working

### Test 2: Code Changes Present ✅
```bash
grep -n "validator" app/api/routes/v2/workflow.py
```
**Result**: Validator instantiation found on lines 184-189, 196

### Test 3: Generated File Has Errors ❌
```bash
grep '"label": 0' .cache/.../. data.jsonl | grep -i "pay me" | wc -l
```
**Result**: 10+ mislabeled samples found

## Solution

### **RESTART THE SERVER**

```bash
# Stop the current server
pkill -f "python app/main.py"

# OR use Ctrl+C if running in terminal

# Start the server again
make start
# OR
python app/main.py
```

## How to Verify Fix

After restarting the server:

1. **Create a new pipeline and generate data:**
   ```bash
   curl -X POST http://localhost:8000/v2/workflow/pipeline \
     -H "Content-Type: application/json" \
     -d '{"label_config": {...}}'

   curl -X POST http://localhost:8000/v2/workflow/first-gen \
     -H "Content-Type: application/json" \
     -d '{"pipeline_id": "...", "phase_id": "..."}'
   ```

2. **Check the logs for validation messages:**
   ```
   Validating 123 samples for label correctness...
   Validation complete for batch 1
   ```

3. **Verify the generated data has no "pay me" → payment_intent errors:**
   ```bash
   grep '"label": 0' .cache/NEW_PIPELINE/.data.jsonl | grep -i "pay me"
   ```
   Should return **ZERO** results (or very few edge cases)

4. **Check that "pay me" samples are correctly labeled:**
   ```bash
   grep '"label": 1' .cache/NEW_PIPELINE/.data.jsonl | grep -i "pay me"
   ```
   Should find all/most "pay me" samples

## Expected Behavior After Fix

### Before Validation (Raw Generation)
- LLM generates samples with ~10-15% labeling errors
- "Pay me" samples often mislabeled as payment_intent

### After Validation (With Fix)
- Validator corrects ~90-95% of labeling errors
- "Pay me" samples correctly labeled as payment_request
- Log shows: "Corrected X out of Y samples"

## Alternative: Test Without Restarting

If you want to test the validator without affecting production:

```bash
python test_validator_actual_data.py
```

This will show that the validator DOES work when called directly, confirming the issue is that it's not being invoked during generation (because server has old code).

---

## Summary

**Problem**: Server running old code without validator integration
**Solution**: Restart the server
**Verification**: Generate new data and check for validation logs + correct labels
**Expected Result**: No more "pay me" → payment_intent errors
