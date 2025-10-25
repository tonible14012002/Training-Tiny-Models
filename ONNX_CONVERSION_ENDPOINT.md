# ONNX Conversion Endpoint

## Overview

A new endpoint has been added to convert trained models to ONNX format for optimized inference deployment.

## Changes Made

### 1. Validator LLM Enhancement (main.py)

Added a dedicated `validator_llm` using GPT-4.1 for improved validation accuracy:

```python
validator_llm = LiteLLMProvider(LLMSettings(
    llm_model_name="gpt-4.1",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.0,  # Deterministic validation
    num_retries=2,
    max_tokens=4096,
))
```

**Benefits:**
- Higher accuracy than GPT-4o-mini for critical validation tasks
- Deterministic results (temperature=0.0)
- Separation of concerns (validator vs data generation)

### 2. ONNX Conversion Endpoint

#### Location
- `app/api/routes/v2/workflow.py:1861-1978`

#### Endpoint
```
POST /v2/workflow/convert-to-onnx
```

#### Request Schema
```json
{
  "model_path": ".checkpoints/pipeline_id/phase_number",
  "output_name": "my_onnx_model"  // optional
}
```

#### Response
- **Content-Type:** `application/zip`
- **Content-Disposition:** `attachment; filename=model_name.zip`
- Returns a downloadable zip file containing the ONNX model and tokenizer

## How It Works

### Conversion Process

1. **Validates Model Path**
   - Checks if the model directory exists
   - Verifies the `_merged` folder is present

2. **Loads Model and Tokenizer**
   - Loads merged model from `{model_path}/_merged`
   - Loads tokenizer from `{model_path}` (parent directory)

3. **Converts to ONNX**
   - Uses `optimum.onnxruntime.ORTModelForSequenceClassification`
   - Exports model with `export=True` parameter

4. **Saves Output**
   - Creates temporary directory: `.cache/onnx/{output_name}_temp/`
   - Saves tokenizer files to root of temp directory
   - Saves ONNX model to `onnx/` subdirectory
   - Creates zip archive for download

5. **Returns Zip File**
   - Automatically triggers browser download
   - Includes proper headers for file download

### Output Files

The zip file contains the following structure:

```
<root>/
  ├── tokenizer_config.json      # Tokenizer settings
  ├── special_tokens_map.json    # Special tokens mapping
  ├── tokenizer.json             # Tokenizer configuration (~711 KB)
  ├── vocab.txt                  # Vocabulary file (~231 KB)
  └── onnx/
      ├── config.json            # Model configuration
      └── model.onnx             # The converted ONNX model (~17.6 MB)
```

**Important:** The ONNX model file is located in the `onnx/` subdirectory, while tokenizer files are at the root level.

## Usage Examples

### Using curl

```bash
curl -X POST http://localhost:8000/v2/workflow/convert-to-onnx \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model_path": ".checkpoints/payment_classification_v2/16",
    "output_name": "payment_model_onnx"
  }' \
  --output payment_model_onnx.zip
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v2/workflow/convert-to-onnx",
    json={
        "model_path": ".checkpoints/pipeline_id/phase_number",
        "output_name": "my_model_onnx"
    },
    headers={"X-API-Key": "your-api-key"}
)

# Save the downloaded file
with open("model.zip", "wb") as f:
    f.write(response.content)
```

### Using the Test Script

```bash
# Start the server
make start

# In another terminal, run the test
python test_onnx_api.py
```

## Loading the ONNX Model

After extracting the zip file, you can load the model like this:

### Python Example

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Assume you extracted the zip to "my_model/"
model_dir = "my_model"

# Load tokenizer from root directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load ONNX model from onnx/ subdirectory
model = ORTModelForSequenceClassification.from_pretrained(f"{model_dir}/onnx")

# Use for inference
text = "pay john $50"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
```

### Directory Structure After Extraction

```
my_model/
  ├── tokenizer_config.json
  ├── special_tokens_map.json
  ├── tokenizer.json
  ├── vocab.txt
  └── onnx/
      ├── config.json
      └── model.onnx
```

## Testing

Two test scripts are provided:

### 1. Standalone Conversion Test
```bash
python test_onnx_conversion.py
```

Tests the conversion logic without requiring the server to be running.

### 2. API Endpoint Test
```bash
python test_onnx_api.py
```

Tests the actual API endpoint (requires server to be running).

## Model Structure Requirements

The model directory must follow this structure:

```
.checkpoints/pipeline_id/phase_number/
├── _merged/                      # Required: merged model
│   ├── config.json
│   └── model.safetensors
├── tokenizer.json                # Required: tokenizer files
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── adapter_model.safetensors     # LoRA adapter (not used for ONNX)
```

## Error Handling

The endpoint returns appropriate error messages for common issues:

### Model Path Not Found
```json
{
  "error": "Model path not found",
  "message": "The model path does not exist: {path}"
}
```

### Missing Merged Model
```json
{
  "error": "Merged model not found",
  "message": "The _merged folder does not exist in: {path}"
}
```

### Conversion Error
```json
{
  "error": "error message",
  "message": "An error occurred during ONNX conversion"
}
```

## Performance Characteristics

- **Conversion Time:** ~10-30 seconds (depends on model size)
- **Output Size:** ~15-18 MB (for BERT-tiny models)
- **Memory Usage:** Moderate (loads full model into memory)

## Integration with Workflow

This endpoint integrates with the existing fine-tuning pipeline:

1. Create pipeline: `POST /workflow/pipeline`
2. Generate data: `POST /workflow/first-gen`
3. Train model: `POST /workflow/train`
4. Evaluate model: `POST /workflow/evaluate`
5. **Convert to ONNX:** `POST /workflow/convert-to-onnx` ← **NEW**

## Schemas Added

### ConvertToONNXRequest
```python
class ConvertToONNXRequest(BaseModel):
    model_path: str  # Path to model checkpoint
    output_name: Optional[str] = None  # Custom output name
```

### ConvertToONNXResponse
```python
class ConvertToONNXResponse(BaseModel):
    message: str
    onnx_path: str
    model_path: str
```

**Note:** The response schema is defined but not used since the endpoint returns a `FileResponse` for direct download.

## Files Modified

1. `app/main.py` - Added `validator_llm` configuration
2. `app/api/routes/v2/workflow.py` - Added ONNX conversion endpoint
3. `app/core/schemas/workflow.py` - Added request/response schemas

## Files Created

1. `test_onnx_conversion.py` - Standalone conversion test
2. `test_onnx_api.py` - API endpoint test
3. `ONNX_CONVERSION_ENDPOINT.md` - This documentation

## Notes

- The endpoint uses `FileResponse` which automatically handles file downloads
- The `Content-Disposition` header triggers browser download
- Temporary files are stored in `.cache/onnx/`
- Existing ONNX directories are cleaned up before conversion
- The conversion uses the optimum library's ONNX Runtime integration
