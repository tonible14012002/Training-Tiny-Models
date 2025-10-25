# ONNX Conversion API Update Summary

## Changes Made

Updated the ONNX conversion endpoint to organize files with the ONNX model in a subdirectory.

## New ZIP Structure

### Before (Old Structure)
```
<root>/
  ├── config.json
  ├── model.onnx
  ├── special_tokens_map.json
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── vocab.txt
```

### After (New Structure)
```
<root>/
  ├── tokenizer_config.json      # Tokenizer settings
  ├── special_tokens_map.json    # Special tokens mapping
  ├── tokenizer.json             # Tokenizer configuration (~711 KB)
  ├── vocab.txt                  # Vocabulary file (~231 KB)
  └── onnx/                      # NEW: ONNX model subdirectory
      ├── config.json            # Model configuration
      └── model.onnx             # The converted ONNX model (~17.6 MB)
```

## Key Changes

### 1. Directory Structure
- **Tokenizer files** are now at the **root level** of the zip
- **ONNX model files** are now in an **`onnx/` subdirectory**

### 2. Code Changes

**File:** `app/api/routes/v2/workflow.py`

**Before:**
```python
onnx_output_dir = Path(".cache/onnx") / output_name
ort_model.save_pretrained(str(onnx_output_dir))
tokenizer.save_pretrained(str(onnx_output_dir))
```

**After:**
```python
temp_output_dir = Path(".cache/onnx") / f"{output_name}_temp"
onnx_subdir = temp_output_dir / "onnx"

# Save tokenizer to root
tokenizer.save_pretrained(str(temp_output_dir))

# Save ONNX to subdirectory
ort_model.save_pretrained(str(onnx_subdir))
```

### 3. Loading the Model

**Before:**
```python
# Everything was in root
tokenizer = AutoTokenizer.from_pretrained("my_model")
model = ORTModelForSequenceClassification.from_pretrained("my_model")
```

**After:**
```python
# Tokenizer at root, ONNX in subdirectory
tokenizer = AutoTokenizer.from_pretrained("my_model")
model = ORTModelForSequenceClassification.from_pretrained("my_model/onnx")
```

## Benefits

1. **Clear Separation**: Tokenizer and model files are logically separated
2. **Better Organization**: ONNX-specific files grouped in their own directory
3. **Flexibility**: Easier to add multiple model formats in the future
4. **Consistency**: Matches common deployment patterns

## Testing

### Test Results

```bash
$ python test_onnx_conversion.py

Verifying saved files in root:
  - onnx/ (directory)
  - special_tokens_map.json
  - tokenizer.json
  - tokenizer_config.json
  - vocab.txt

Verifying ONNX files in onnx/ subdirectory:
  - onnx/config.json
  - onnx/model.onnx

✓ ONNX conversion successful!
✓ Zip file: .cache/onnx/test_onnx_model.zip (15.82 MB)

Verifying zip file structure:
  onnx/
  onnx/config.json
  onnx/model.onnx
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  vocab.txt

✓ ONNX model loaded and tested successfully!
```

### Zip File Verification

```bash
$ unzip -l .cache/onnx/test_onnx_model.zip

Archive:  .cache/onnx/test_onnx_model.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
        0  10-25-2025 18:32   onnx/
     1491  10-25-2025 18:32   tokenizer_config.json
      695  10-25-2025 18:32   special_tokens_map.json
   711661  10-25-2025 18:32   tokenizer.json
   231508  10-25-2025 18:32   vocab.txt
      841  10-25-2025 18:32   onnx/config.json
 17605651  10-25-2025 18:32   onnx/model.onnx
---------                     -------
 18551847                     7 files
```

## Usage Example

### Extract and Load

```bash
# Download from API
curl -X POST http://localhost:8000/v2/workflow/convert-to-onnx \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"model_path": ".checkpoints/pipeline_id/phase_number"}' \
  --output model.zip

# Extract
unzip model.zip -d my_model

# Directory structure
my_model/
  ├── tokenizer_config.json
  ├── special_tokens_map.json
  ├── tokenizer.json
  ├── vocab.txt
  └── onnx/
      ├── config.json
      └── model.onnx
```

### Python Inference

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Load tokenizer from root
tokenizer = AutoTokenizer.from_pretrained("my_model")

# Load ONNX model from onnx/ subdirectory
model = ORTModelForSequenceClassification.from_pretrained("my_model/onnx")

# Run inference
text = "pay john $50"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

print(f"Prediction: {predictions.item()}")
```

## Files Modified

1. **app/api/routes/v2/workflow.py** (lines 1916-1966)
   - Updated directory structure creation
   - Save tokenizer to root directory
   - Save ONNX model to `onnx/` subdirectory

2. **test_onnx_conversion.py**
   - Updated to match new directory structure
   - Added zip file structure verification
   - Updated model loading paths

3. **ONNX_CONVERSION_ENDPOINT.md**
   - Updated output file structure documentation
   - Added loading example with new structure
   - Updated conversion process description

## Backward Compatibility

⚠️ **Breaking Change**: This is a breaking change for users who expect the old structure.

**Migration Guide:**
- Old code: `model = ORTModelForSequenceClassification.from_pretrained("my_model")`
- New code: `model = ORTModelForSequenceClassification.from_pretrained("my_model/onnx")`

**Recommendation:** Update your inference code to use the new path structure.

## Future Enhancements

Possible future additions:
1. Multiple model formats (TorchScript, TFLite, etc.) in separate subdirectories
2. Model metadata file at root level
3. Configuration files for deployment platforms
4. Example inference code included in zip

## Summary

✅ ONNX model now in `onnx/` subdirectory
✅ Tokenizer files at root level
✅ Better organization and separation
✅ All tests passing
✅ Documentation updated
✅ Ready for deployment
