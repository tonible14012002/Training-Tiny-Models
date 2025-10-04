# API Documentation

## Base URL
`http://localhost:8000`

---

## Health & Status

### `GET /`
Root endpoint with API information.

**Response:**
```json
{
  "message": "Fine-tuning Workflow API",
  "version": "1.0.0"
}
```

### `GET /health-check`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "environment": "dev"
}
```

---

## Data Generation

### `POST /workflow/generate-data-v2`
Generate synthetic training data using DataGeneratorV2 with parallel processing.

**Response:**
```json
{
  "message": "Data generation v2 started with parallel processing",
  "status": "in_progress"
}
```

### `POST /workflow/fresh-generate-eval`
Generate fresh evaluation dataset (intent and open intent samples).

**Response:**
```json
{
  "message": "Evaluation data generation completed",
  "status": "completed",
  "iteration_number": 1,
  "intent_samples_generated": 50,
  "open_intent_messages_generated": 20,
  "total_generated": 70
}
```

### `POST /workflow/generate-fix-data`
Generate synthetic data using custom prompt for fixing model issues.

**Request Body:**
```json
{
  "prompt": "string",
  "amount": 100  // optional
}
```

**Response:**
```json
{
  "message": "Fix data generation completed. Generated 100 samples, 85 saved after deduplication.",
  "status": "completed",
  "samples_generated": 100,
  "samples_saved": 85,
  "dataset_path": ".cache/.data_fix_gen_20231002_181035.jsonl",
  "main_file": ".cache/.data.jsonl"
}
```

---

## Model Training

### `POST /workflow/train`
Train model from scratch with specified inference type.

**Query Parameters:**
- `inference_type` (string): `"adb"` or `"prob"` (default: `"prob"`)

**Response:**
```json
{
  "status": "completed",
  "checkpoint_number": 1,
  "inference_type": "prob"
}
```

### `POST /workflow/continual-train`
Continue training from an existing checkpoint with sub-versioning.

**Query Parameters:**
- `checkpoint_id` (string, required): Checkpoint identifier (e.g., `"10"`, `"10.1"`)
- `inference_type` (string): `"adb"` or `"prob"` (default: `"prob"`)
- `dataset_path` (string, optional): Path to specific dataset file

**Response:**
```json
{
  "status": "completed",
  "checkpoint_id": "10.1",
  "base_checkpoint": "10",
  "inference_type": "prob",
  "dataset_path": ".cache/.data.jsonl",
  "message": "Continual training completed. New checkpoint: 10.1"
}
```

---

## Model Evaluation & Analysis

### `POST /workflow/evaluate`
Evaluate model performance with comprehensive analysis.

**Request Body:**
```json
{
  "checkpoint_id": "1",  // optional, defaults to latest
  "iteration_number": 1,  // optional, defaults to latest
  "include_test_cases": false,
  "include_open_intent": true
}
```

**Response:**
```json
{
  "message": "Model evaluation completed successfully",
  "status": "completed",
  "checkpoint_path": ".checkpoints/1",
  "checkpoint_id": "1",
  "evaluation_data_info": {
    "iteration_number": 1,
    "known_intent_samples": 100,
    "open_intent_samples": 20,
    "total_samples": 120
  },
  "results": {
    "overall_metrics": {
      "accuracy": 0.95,
      "macro_f1": 0.93,
      "macro_precision": 0.94,
      "macro_recall": 0.92,
      "total_samples": 100,
      "correct_predictions": 95,
      "unknown_predictions": 0,
      "unknown_rate": 0.0
    },
    "per_label_metrics": {
      "payment_intent": {
        "accuracy": 0.96,
        "precision": 0.95,
        "recall": 0.94,
        "f1_score": 0.945,
        "samples": 50,
        "true_positives": 47,
        "false_positives": 2,
        "true_negatives": 48,
        "false_negatives": 3
      }
    },
    "open_intent_analysis": {
      "total_samples": 20,
      "detected_as_unknown": 18,
      "unknown_rate": 0.9,
      "false_positive_rate": 0.1,
      "misclassified": []
    },
    "adb_info": null,
    "error_patterns": {}
  }
}
```

### `POST /workflow/analyze-error-patterns`
Analyze error patterns using LLM to identify issues and generate recommendations.

**Request Body:**
```json
{
  "checkpoint_id": "1",  // optional, defaults to latest
  "iteration_number": 1   // optional, defaults to latest
}
```

**Response:**
```json
{
  "message": "Error pattern analysis completed. Analyzed 3 error patterns.",
  "status": "completed",
  "checkpoint_path": ".checkpoints/1",
  "checkpoint_id": "1",
  "iteration_number": 1,
  "error_analyses": [
    {
      "predicted_label": "payment_intent",
      "expected_label": "payment_request",
      "identified_issues": [
        "Missing examples with pronoun 'I' as recipient",
        "Overuse of keyword 'pay' in payment_request examples"
      ],
      "data_actions": [
        {
          "label": "payment_request",
          "keywords_to_include": ["I", "me", "my"],
          "keywords_to_avoid": ["pay"],
          "sentence_patterns": [],
          "context_requirements": [],
          "diversity_constraints": [],
          "ignore": false
        }
      ]
    }
  ],
  "evaluation_summary": {
    "overall_accuracy": 0.95,
    "macro_f1": 0.93,
    "total_errors": 5
  }
}
```

---

## Model Inference

### `POST /workflow/inference`
Run inference on text samples using a trained model.

**Query Parameters:**
- `checkpoint_id` (string, optional): Checkpoint identifier (defaults to latest)

**Request Body:**
```json
{
  "text": ["send me money", "I will pay you tomorrow"]
}
```

**Response:**
```json
{
  "message": "Inference completed",
  "status": "completed",
  "results": {
    "checkpoint": ".checkpoints/1",
    "checkpoint_id": "1",
    "inference_type": "prob",
    "predictions": [
      {
        "text": "send me money",
        "predicted_label": "payment_request"
      },
      {
        "text": "I will pay you tomorrow",
        "predicted_label": "payment_intent"
      }
    ],
    "model_info": {
      "model_type": "bert-tiny",
      "num_labels": 3
    }
  }
}
```

---

## Prompt Building

### `POST /workflow/make-fix-prompt`
Generate prompt for targeted data generation based on error patterns.

**Request Body:**
```json
{
  "actions": [
    {
      "label": "payment_request",
      "keywords_to_include": ["I", "me"],
      "keywords_to_avoid": [],
      "sentence_patterns": ["[pronoun] need money"],
      "context_requirements": ["casual conversation"],
      "diversity_constraints": ["vary sentence length"],
      "ignore": false
    }
  ]
}
```

**Response:**
```json
{
  "message": "Prompt generated successfully",
  "status": "completed",
  "prompt": "Generate examples for payment_request with keywords: I, me..."
}
```

---

## Label Configuration

### Payment Classification V2 Labels
```
payment_request (0): User requesting to receive money
payment_intent (1): User declaring they will send money or instructing system to execute payment
open_intent (2): Arbitrary chat messages unrelated to payment
```

---

---

## Orchestrator Pipeline (NEW)

### `POST /workflow/run-orchestrator`
Run the complete iterative training orchestration pipeline with configurable parameters.

**What it does:**
1. Loads model from initial checkpoint
2. Evaluates on evaluation dataset
3. Analyzes errors using LLM (error pattern analyzer)
4. Generates targeted training samples based on error analysis
5. Continues training from previous checkpoint (sub-versioning)
6. Repeats until all labels achieve target F1 or max iterations reached

**Request Body:**
```json
{
  "initial_checkpoint_id": "11.7",    // Starting checkpoint (default: "11.7")
  "max_iterations": 20,               // Max iterations (default: 20, range: 1-100)
  "target_f1_per_label": 0.7,         // Target F1 for ALL labels (default: 0.7, range: 0.0-1.0)
  "samples_per_action": 500,          // Samples per action (default: 500, range: 10-2000)
  "iteration_number": null            // Eval dataset iteration (default: null = latest)
}
```

**Success Response:**
```json
{
  "success": true,
  "status": "completed",
  "termination_reason": "All labels achieved target F1 >= 0.7",
  "iterations_completed": 5,
  "final_checkpoint": "11.12",
  "final_metrics": {
    "overall_accuracy": 0.89,
    "macro_f1": 0.85,
    "per_label_f1": {
      "payment_request": 0.82,
      "payment_intent": 0.88,
      "open_intent": 0.85
    }
  },
  "config": {
    "target_f1_per_label": 0.7,
    "samples_per_action": 500,
    "initial_checkpoint_id": "11.7"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "status": "error",
  "message": "Orchestrator run failed: Checkpoint 11.7 not found",
  "error": "Checkpoint 11.7 not found",
  "config": {
    "initial_checkpoint_id": "11.7",
    "max_iterations": 20,
    "target_f1_per_label": 0.7,
    "samples_per_action": 500,
    "iteration_number": null
  }
}
```

**Usage Examples:**

Basic request (use all defaults):
```bash
curl -X POST "http://localhost:8000/workflow/run-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Custom configuration:
```bash
curl -X POST "http://localhost:8000/workflow/run-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_checkpoint_id": "11.7",
    "max_iterations": 30,
    "target_f1_per_label": 0.75,
    "samples_per_action": 600
  }'
```

Quick test (low samples, few iterations):
```bash
curl -X POST "http://localhost:8000/workflow/run-orchestrator" \
  -H "Content-Type: application/json" \
  -d '{
    "max_iterations": 5,
    "target_f1_per_label": 0.6,
    "samples_per_action": 100
  }'
```

**Convergence Logic:**
- Pipeline stops when **ALL labels** achieve F1 >= `target_f1_per_label`
- Even if 2 out of 3 labels meet target, continues until all 3 do

**Checkpoint Versioning:**
- Start: `11.7`
- After iteration 1: `11.8`
- After iteration 2: `11.9`
- After iteration 3: `11.10`

**Prerequisites:**
1. Checkpoint exists: `initial_checkpoint_id` must exist
2. Evaluation dataset ready: Run `/workflow/fresh-generate-eval` first
3. Human seeds available: `.cache/human_seed.json` must exist

**Recommended Configurations:**

Development/Testing:
```json
{
  "max_iterations": 5,
  "target_f1_per_label": 0.6,
  "samples_per_action": 100
}
```

Production:
```json
{
  "max_iterations": 30,
  "target_f1_per_label": 0.75,
  "samples_per_action": 500
}
```

High-Quality Training:
```json
{
  "max_iterations": 50,
  "target_f1_per_label": 0.8,
  "samples_per_action": 1000
}
```

---

## Error Response Format
All endpoints return errors in this format:
```json
{
  "message": "Error description",
  "status": "error"
}
```
