# TypeScript Schema Definition

This file contains TypeScript interfaces that correspond to the Pydantic models defined in the Python codebase.

## Table of Contents

- [Database Models](#database-models)
- [Workflow Schemas](#workflow-schemas)
- [Analysis Schemas](#analysis-schemas)
- [Orchestrator Schemas](#orchestrator-schemas)
- [Inference Schemas](#inference-schemas)

---

## Database Models

### Pipeline

```typescript
interface Pipeline {
  id: string;
  name: string;
  created_at: string; // ISO 8601 datetime
  updated_at: string; // ISO 8601 datetime
}
```

### LabelConfig

```typescript
interface LabelConfig {
  id: string;
  pipeline_id: string;
  name: string;
  id2label: string; // JSON string: {"0": "payment_request", "1": "payment_intent", "2": "open_intent"}
  label2id: string; // JSON string: {"payment_request": 0, "payment_intent": 1, "open_intent": 2}
  label_explanation: string | null; // JSON string
  created_at: string; // ISO 8601 datetime
}
```

### PipelinePhase

```typescript
interface PipelinePhase {
  id: string;
  pipeline_id: string;
  previous_phase_id: string | null;
  phase_number: number;
  checkpoint_id: string | null;
  checkpoint_path: string | null;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  created_at: string; // ISO 8601 datetime
  completed_at: string | null; // ISO 8601 datetime
}
```

### ComposalDataset

```typescript
interface ComposalDataset {
  id: string;
  pipeline_id: string;
  phase_id: string | null; // Phase that generated this composal dataset
  name: string | null;
  description: string | null;
  file_path: string | null; // Path to the combined dataset file
  total_samples: number;
  created_at: string; // ISO 8601 datetime
}
```

### DatasetFile

```typescript
interface DatasetFile {
  id: string;
  parent_dataset_id: string;
  file_path: string;
  phase_id: string;
  file_type: 'train' | 'validation' | 'test' | 'generated' | null;
  sample_count: number;
  created_at: string; // ISO 8601 datetime
}
```

### ErrorBucket

```typescript
interface ErrorBucket {
  id: string;
  pipeline_id: string;
  name: string;
  description: string;
  examples: string; // JSON serialized list of Sample objects
  data_generation_strategy: string | null;
  created_at: string; // ISO 8601 datetime
  updated_at: string; // ISO 8601 datetime
}
```

### PhaseErrorBucket

```typescript
interface PhaseErrorBucket {
  id: string;
  phase_id: string;
  bucket_id: string;
  error_count: number;
  generation_count: number;
  examples: string; // JSON serialized list of Sample objects
  created_at: string; // ISO 8601 datetime
  updated_at: string; // ISO 8601 datetime
}
```

### ErrorBucketPhase

```typescript
interface ErrorBucketPhase {
  id: string;
  phase_id: string;
  base_error_bucket_id: string; // The base bucket this was cloned from
  name: string;
  description: string;
  examples: string; // JSON list of examples with labels
  count: number; // Number of errors in this category for this phase
  data_generation_strategy: string | null;
  created_at: string; // ISO 8601 datetime
  updated_at: string; // ISO 8601 datetime
}
```

### HumanTestSet

```typescript
interface HumanTestSet {
  id: string;
  pipeline_id: string | null;
  name: string;
  file_path: string;
  description: string | null;
  sample_count: number;
  created_at: string; // ISO 8601 datetime
}
```

### TrainedModel

```typescript
interface TrainedModel {
  id: string;
  phase_id: string;
  model_name: string;
  model_save_path: string;
  training_time: number | null; // Training time in seconds
  dataset_file_id: string | null;
  training_params: string | null; // JSON string with training parameters
  status: 'training' | 'completed' | 'failed';
  created_at: string; // ISO 8601 datetime
  completed_at: string | null; // ISO 8601 datetime
}
```

### EvaluationResult

```typescript
interface EvaluationResult {
  id: string;
  trained_model_id: string;
  human_test_set_id: string | null;
  dataset_file_id: string | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  label_metrics: string | null; // JSON string for per-label metrics
  metrics: string | null; // JSON string for additional metrics
  evaluated_at: string; // ISO 8601 datetime
}
```

---

## Workflow Schemas

### Sample

```typescript
interface Sample {
  msg: string;
  label: string | number; // Flexible to support different label types
}
```

### Result

```typescript
interface Result {
  messages: Sample[];
}
```

### EvaluationRequest

```typescript
interface EvaluationRequest {
  iteration_number?: number | null;
  include_test_cases?: boolean;
  include_open_intent?: boolean;
  checkpoint_id?: string | null; // e.g., "1", "2", "1.1", "2.3"
  threshold_config?: {
    thresholds: {
      [label: string]: number; // e.g., { "payment_intent": 0.75, "payment_request": 0.70 }
    };
    fallback_label: string;
  } | null;
}
```

### EvaluationResponse

```typescript
interface EvaluationResponse {
  message: string;
  status: string;
  checkpoint_path: string;
  checkpoint_id?: string | null; // e.g., "1", "2", "1.1", "2.3"
  evaluation_data_info: Record<string, any>;
  results: Record<string, any>;
}
```

### FixGenRequest

```typescript
interface FixGenRequest {
  prompt: string;
  amount?: number | null;
}
```

### AnalyzeErrorPatternsRequest

```typescript
interface AnalyzeErrorPatternsRequest {
  checkpoint_id?: string | null; // e.g., "1", "2", "1.1", "2.3"
  iteration_number?: number | null;
}
```

### OrchestratorRunRequest

```typescript
interface OrchestratorRunRequest {
  initial_checkpoint_id?: string; // Default: "11.7"
  max_iterations?: number; // Default: 20, min: 1, max: 100
  target_f1_per_label?: number; // Default: 0.7, min: 0.0, max: 1.0
  samples_per_action?: number; // Default: 500, min: 10, max: 2000
  iteration_number?: number | null;
}
```

### LabelConfigRequest

```typescript
interface LabelConfigRequest {
  name: string;
  id2label: Record<string, string>; // e.g., {"0": "payment_request", "1": "payment_intent"}
  label2id: Record<string, number>; // e.g., {"payment_request": 0, "payment_intent": 1}
  label_explanation?: Record<string, string> | null;
}
```

### CreatePipelineRequest

```typescript
interface CreatePipelineRequest {
  name: string;
  label_config: LabelConfigRequest;
}
```

### LabelConfigResponse

```typescript
interface LabelConfigResponse {
  id: string;
  name: string;
  id2label: Record<string, string>;
  label2id: Record<string, number>;
  label_explanation?: Record<string, string> | null;
  created_at: string;
}
```

### PipelineResponse

```typescript
interface PipelineResponse {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  label_config?: LabelConfigResponse | null;
}
```

### ListPipelinesResponse

```typescript
interface ListPipelinesResponse {
  message: string;
  data: PipelineResponse[];
}
```

### ClassifyErrorRequest

```typescript
interface ClassifyErrorRequest {
  phase_id?: string | null;
}
```

### StartPhaseRequest

```typescript
interface StartPhaseRequest {
  pipeline_id: string;
  phase_id?: string | null; // Optional phase ID to resume
}
```

### TestTrainPhaseRequest

```typescript
interface TestTrainPhaseRequest {
  phase_id: string;
  ds_file_path: string;
  checkpoint_path?: string; // Default: ".checkpoints"
  cache_path?: string; // Default: ".cache"
}
```

### StartTrainPhase

```typescript
interface StartTrainPhase {
  phase_id: string;
}
```

### StartEvaluationPhase

```typescript
interface StartEvaluationPhase {
  phase_id: string;
  confidence_thresholds?: number; // Default: 0.5
}
```

### StartErrBucketPhase

```typescript
interface StartErrBucketPhase {
  phase_id: string;
}
```

### TestEvaluationRequest

```typescript
interface TestEvaluationRequest {
  model_path: string;
  pipeline_id: string;
  cache_path?: string; // Default: ".cache"
}
```

### TestFirstGenRequest

```typescript
interface TestFirstGenRequest {
  pipeline_id: string;
  cache_path?: string; // Default: ".cache"
}
```

### PHASE_STATUS

```typescript
const PHASE_STATUS = {
  NOT_STARTED: 'not_started',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  FAILED: 'failed',
} as const;

type PhaseStatus = typeof PHASE_STATUS[keyof typeof PHASE_STATUS];
```

---

## Analysis Schemas

### CategorizeResult

```typescript
interface CategorizeResult {
  bucket: string; // The name of the error bucket
  reason: string; // Brief explanation of why the test case was categorized into this bucket
}
```

### AnalyzeTestCase

```typescript
interface AnalyzeTestCase {
  sample: Sample;
  predicted: string;
  prob: number;
}
```

### ErrorBucketSchema

```typescript
interface ErrorBucketSchema {
  name: string; // Unique name that briefly summarizes the error type
  description: string; // Detailed description of the error bucket
  examples?: Sample[];
  data_generation_strategy?: string | null; // Strategy for generating additional data samples
}
```

### LLMErrorAnalysis

```typescript
interface LLMErrorAnalysis {
  error_buckets?: string[]; // List of error bucket names
}
```

---

## Orchestrator Schemas

### IterationMetrics

```typescript
interface IterationMetrics {
  iteration: number;
  accuracy: number; // 0.0 to 1.0
  macro_f1: number; // 0.0 to 1.0
  unknown_rate: number; // 0.0 to 1.0
  total_samples: number;
  checkpoint_path: string;
  timestamp: number; // Unix timestamp
  training_time: number; // Seconds
  evaluation_time: number; // Seconds
}
```

### PipelineConfig

```typescript
interface PipelineConfig {
  max_iterations?: number; // Default: 10, min: 1, max: 100
  target_accuracy?: number; // Default: 0.85, min: 0.0, max: 1.0
  target_macro_f1?: number; // Default: 0.80, min: 0.0, max: 1.0
  early_termination_threshold?: number; // Default: 0.02, min: 0.0, max: 0.5
  min_improvement_iterations?: number; // Default: 2, min: 1, max: 10
  data_generation_batch_size?: number; // Default: 15, min: 1, max: 100
}
```

### PipelineStatus

```typescript
interface PipelineStatus {
  is_running?: boolean; // Default: false
  current_iteration?: number; // Default: 0
  total_iterations?: number; // Default: 0
  last_metrics?: IterationMetrics | null;
  best_metrics?: IterationMetrics | null;
  termination_reason?: string | null;
  start_time?: number | null; // Unix timestamp
  metrics_history?: IterationMetrics[];
}
```

### PipelineResult

```typescript
interface PipelineResult {
  success: boolean;
  status: string;
  termination_reason: string;
  total_iterations: number;
  total_time: number; // Seconds
  best_metrics?: Record<string, any> | null;
  final_metrics?: Record<string, any> | null;
  metrics_history?: Record<string, any>[];
  error?: string | null;
}
```

### PipelineStatusResponse

```typescript
interface PipelineStatusResponse {
  message: string;
  status: string;
  is_running: boolean;
  current_iteration: number;
  total_iterations: number;
  termination_reason?: string | null;
  last_metrics?: Record<string, any> | null;
  best_metrics?: Record<string, any> | null;
  metrics_history?: Record<string, any>[];
}
```

### PipelineActionResponse

```typescript
interface PipelineActionResponse {
  success: boolean;
  message: string;
  iterations_completed?: number | null;
  error?: string | null;
}
```

---

## Inference Schemas

### ThresholdConfig

```typescript
interface ThresholdConfig {
  thresholds: Record<string, number>; // Label to minimum probability threshold mapping
  fallback_label?: string; // Default: "Unknown"
}
```

### InferenceRequest

```typescript
interface InferenceRequest {
  text: string[];
}
```

---

## Notes

1. **Datetime Fields**: All datetime fields in Python are represented as ISO 8601 strings in TypeScript (e.g., `"2025-10-13T10:30:00Z"`).

2. **JSON String Fields**: Some fields store JSON as strings in the database. When working with these in TypeScript, you'll need to parse them:
   - `id2label`, `label2id`, `label_explanation` in `LabelConfig`
   - `examples` in `ErrorBucket`, `PhaseErrorBucket`
   - `training_params` in `TrainedModel`
   - `metrics`, `label_metrics` in `EvaluationResult`

3. **Optional Fields**: Fields marked with `| null` or `?` are optional and may not be present in the response.

4. **Enums**: Status fields use string literals for type safety. Consider using TypeScript's `as const` for enum-like behavior.

5. **Label Types**: The `label` field in `Sample` can be either a `string` or `number` depending on context.

6. **Foreign Keys**: Fields ending in `_id` reference the `id` field of another entity.

## Example Usage

```typescript
// Creating a new pipeline
const createPipelineRequest: CreatePipelineRequest = {
  name: "Payment Classification Pipeline",
  label_config: {
    name: "Payment Classification v2",
    id2label: {
      "0": "payment_request",
      "1": "payment_intent",
      "2": "open_intent"
    },
    label2id: {
      "payment_request": 0,
      "payment_intent": 1,
      "open_intent": 2
    },
    label_explanation: {
      "payment_intent": "User intends to send/pay money",
      "payment_request": "User requests to receive money",
      "open_intent": "General chat messages"
    }
  }
};

// Starting a new phase
const startPhaseRequest: StartPhaseRequest = {
  pipeline_id: "uuid-string-here"
};

// Evaluation request with thresholds
const evaluationRequest: EvaluationRequest = {
  checkpoint_id: "1.1",
  include_test_cases: true,
  threshold_config: {
    thresholds: {
      "payment_intent": 0.75,
      "payment_request": 0.70,
      "open_intent": 0.60
    },
    fallback_label: "Unknown"
  }
};
```
