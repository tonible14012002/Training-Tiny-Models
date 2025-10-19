# Pipeline Stages Documentation

This document describes each stage in the current pipeline implementation, including what each stage does, what it stores, what it outputs, and the conditions for execution.

---

## 1. First-Gen (Initial Data Generation)

### Purpose
Generate the initial training dataset for a new pipeline phase using LLM-based synthetic data generation.

### Execution Conditions
- A pipeline with label configuration must exist
- Human seed samples must be available
- No prior dataset exists for phase 0

### Process Flow
1. **Phase Creation**: Creates a new phase with `phase_number=0` and status `in_progress`
2. **Database Preparation**: Creates composal dataset and dataset file records before generation begins
3. **Data Generation Loop**:
   - Uses human seed examples as initial inputs
   - Generates samples in parallel batches (configurable parallelization)
   - Each batch generates ~30 samples per API call
   - Continues until target quantity per label is reached
   - Applies deduplication using ROUGE similarity thresholds
   - Each batch is saved incrementally to temp files
4. **Batch Tracking**: After each batch generation, saves batch metadata to database via callback
5. **Final Consolidation**: Copies final dataset to composal directory and updates dataset file status to `done`

### What It Stores
- **Phase Record**: Phase metadata with status and phase number
- **Composal Dataset**: Parent dataset record for the phase
- **Dataset File**: Main dataset file record with file path and sample count
- **Batch Files**: Individual batch file records for tracking incremental generation
- **Physical Files**:
  - Temporary batch JSONL files (one per generation iteration)
  - Final composal dataset JSONL file

### What It Outputs
- Generated sample count
- Composal dataset metadata
- Dataset file information with path and status
- List of batch files with sample counts

### Key Parameters
- `expect_total_each_label`: Dictionary specifying target count per label
- `human_seeds`: Initial seed samples for generation
- `rouge_threshold`: Default 0.6 for deduplication
- `batch_size`: Parallel generations per iteration (5-20 based on target)

---

## 2. Train

### Purpose
Train a text classification model using the generated dataset from First-Gen stage.

### Execution Conditions
- A phase must exist
- Dataset file from First-Gen must be completed (status = `done`)
- Valid label configuration must be present

### Process Flow
1. **Dataset Loading**: Loads the dataset file from the phase's composal dataset
2. **Model Setup**:
   - Loads base model (e.g., `prajjwal1/bert-tiny`)
   - Applies LoRA (Low-Rank Adaptation) configuration for efficient fine-tuning
   - Configures training arguments (learning rate, batch size, epochs)
3. **Data Preprocessing**:
   - Shuffles dataset with random seed
   - Tokenizes text messages with max_length=128
   - Applies padding and truncation
4. **Training Execution**:
   - Trains model for 3 epochs with learning rate 2e-5
   - Uses gradient accumulation (effective batch size = 32)
   - Applies warmup ratio of 0.15
5. **Model Saving**:
   - Saves LoRA adapters to checkpoint directory
   - Merges and saves full model to `_merged` subdirectory
   - Saves inference configuration metadata
6. **Database Recording**: Creates trained model record with checkpoint path

### What It Stores
- **Trained Model Record**: Database record with model path, status, and dataset reference
- **Physical Files**:
  - LoRA adapter weights
  - Merged model weights
  - Tokenizer configuration
  - Training outputs and logs
  - `inference_config.json` with inference type metadata

### What It Outputs
- Checkpoint path (e.g., `.checkpoints/{pipeline_id}/{phase_number}`)
- Trained model metadata including:
  - Model name with version
  - Model save path
  - Associated dataset file ID
  - Training status

### Key Parameters
- `base_model`: Base transformer model identifier
- `lora_config`:
  - Rank (r=16)
  - Alpha (32)
  - Dropout (0.1)
  - Target modules: query, value, dense
- `training_args`:
  - Learning rate: 2e-5
  - Epochs: 3
  - Gradient accumulation steps: 4

---

## 3. Evaluate

### Purpose
Evaluate the trained model on both frozen test set and training pool to identify errors and low-confidence predictions.

### Execution Conditions
- A trained model must exist for the phase
- Frozen test set must be available in cache
- Valid label configuration must be present

### Process Flow
1. **Test Set Evaluation**:
   - Loads frozen test set from `.cache/frozen_test_set.json`
   - Loads trained model from checkpoint path
   - Runs inference on all test samples
   - Calculates overall metrics (accuracy, precision, recall, F1)
   - Calculates per-label metrics
   - Identifies error samples (incorrect predictions)
   - Identifies low-confidence correct predictions
2. **Training Pool Analysis**:
   - Loads training dataset (composal dataset)
   - Shuffles with random seed
   - Runs inference to find low-confidence samples
   - Limits to 200 low-confidence samples with probability < 0.5
3. **Metrics Aggregation**:
   - Combines test set metrics with training pool low-confidence samples
   - Stores comprehensive metrics in database

### What It Stores
- **Evaluation Result Record**: Database record with:
  - Reference to trained model
  - Reference to dataset file used
  - Overall metrics (accuracy, precision, recall, F1)
  - Per-label metrics (F1, precision, recall per class)
  - Error samples with predictions and probabilities
  - Low-confidence correct samples from both test and training sets
- **Label Metrics JSON** containing:
  - Per-label performance statistics
  - Error sample details
  - Low-confidence sample details from training pool

### What It Outputs
- Evaluation metrics summary:
  - Overall accuracy, precision, recall, F1
  - Per-label metrics breakdown
  - Error sample count and details
  - Low-confidence sample count and details
- Full evaluation record with all metrics and samples

### Key Parameters
- `confidence_threshold`: Default 0.5 for identifying low-confidence predictions
- `low_conf_limit`: Maximum 200 low-confidence samples from training pool
- `get_error_samples`: Boolean flag to collect error details
- `get_low_confidence`: Boolean flag to collect low-confidence samples

---

## 4. Generate Fix (Error-Based Data Generation)

### Purpose
Generate targeted training data to address specific error patterns and low-confidence predictions identified during evaluation.

### Current Status
**Partially Implemented** - The error analysis and preparation phase is complete, but the actual data generation is not yet implemented.

### Execution Conditions
- Evaluation results must exist for the phase
- Error samples must be present
- Training pool must be available
- Error buckets must be defined for the pipeline

### Process Flow (Current Implementation)
1. **Error Analysis**:
   - Loads evaluation results for the phase
   - Extracts error samples from test set evaluation
   - Categorizes errors into predefined error buckets using LLM
2. **Error Categorization**:
   - Uses ErrorCategorizer service with LLM
   - Maps each error to specific error bucket
   - Tracks error count per bucket
   - Saves example errors per bucket (up to 100 examples)
3. **Low-Confidence Sample Collection**:
   - Extracts low-confidence samples from training pool
   - Groups by label to ensure balanced representation
   - Targets equal quantity per label
   - Fills gaps with random samples if needed
4. **Generation Configuration Preparation**:
   - Calculates total fix samples needed (default 300)
   - Distributes equally across error buckets
   - Prepares configuration for random generation per label
   - Prepares configuration for low-confidence-based generation

### What It Stores (Current)
- **Phase Error Bucket Records**: Database records linking phase to error buckets with:
  - Error count per bucket
  - Example samples per bucket (up to 100)
  - Bucket metadata
- **Error Categorization Results**:
  - Categorized error testcases
  - Bucket assignments
  - Error patterns identified

### What It Outputs (Current)
- Error bucket statistics:
  - Error count per bucket
  - Example errors per bucket
- Generation configuration:
  - Random samples needed per label
  - Low-confidence samples available per label
  - Samples needed per error bucket
- Low-confidence sample statistics

### What Is Missing (Not Yet Implemented)
- **Actual Data Generation**: Using error buckets and low-confidence samples to generate targeted training data
- **Prompt Building**: Creating specialized prompts for each error bucket
- **Fix Dataset Creation**: Generating and saving fix dataset
- **Integration with Training**: Merging fix dataset with existing training pool

### Key Parameters
- `expect_error_fix`: Target 300 fix samples
- `each_bucket_need`: Minimum 50 samples per error bucket
- `expect_low_conf_each_label`: Balanced distribution across labels
- `low_confidence_threshold`: 0.4 for selecting very uncertain samples

---

## Stage Dependencies

```
First-Gen (Phase 0)
    ↓
    Creates: Composal Dataset + Dataset Files
    ↓
Train
    ↓
    Uses: Dataset File from First-Gen
    Creates: Trained Model + Checkpoint
    ↓
Evaluate
    ↓
    Uses: Trained Model + Frozen Test Set + Training Pool
    Creates: Evaluation Results + Error Samples + Low-Confidence Samples
    ↓
Generate Fix (Partial)
    ↓
    Uses: Error Samples + Low-Confidence Samples + Error Buckets
    Creates: Error Categorization + Generation Config
    (Not Yet: Fix Dataset)
```

---

## Pipeline State Management

### Phase Status Values
- `not_started`: Phase created but not yet executed
- `in_progress`: Phase is currently executing
- `completed`: Phase execution finished successfully
- `failed`: Phase execution encountered an error

### Dataset File Status Values
- `generating`: Data generation in progress (First-Gen)
- `done`: Dataset generation complete and file ready

### Typical Phase Lifecycle
1. Create Pipeline with Label Config
2. Start First-Gen → Phase 0 created with status `in_progress`
3. First-Gen completes → Dataset file status `done`
4. Start Train → Trained model created
5. Start Evaluate → Evaluation results created
6. Classify Errors → Error buckets populated
7. Generate Fix → (Not yet implemented)

---

## Notes

- **ErrBucket (Error Buckets)**: While mentioned to be ignored for now, the error bucket infrastructure is implemented and functional for error categorization. It's ready for use when Generate Fix is fully implemented.

- **Incremental Generation**: First-Gen uses incremental batch saving to track progress and enable recovery if generation is interrupted.

- **Deduplication**: All generated data goes through ROUGE-based deduplication (default threshold 0.6) to ensure dataset diversity.

- **Model Architecture**: Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning of transformer models, reducing computational requirements while maintaining performance.
