# Pipeline Improvement Plan

**Date**: 2025-10-13
**Status**: Planning Phase
**Goal**: Simplify pipeline to enable experimentation and improve model quality visibility

---

## Current State Problems

1. **Lack of Observability**: Pipeline runs without clear visibility into what's happening at each step
2. **Unknown Quality Factors**: Unclear what makes a good dataset, how many examples are enough, or how continual training affects results
3. **Complex Error Bucket System**: Error bucket implementation adds complexity before validating basic pipeline functionality
4. **No Quality Gates**: Pipeline continues even when producing bad models
5. **Limited Experiment Tracking**: Difficult to compare different runs and learn from experiments

---

## Recommended Simplified Pipeline Architecture

### Phase 1: Basic Observable Pipeline (MVP)

**Goal**: Create a minimal working pipeline where every step is visible, measurable, and trackable.

#### Core Components

#### 1. Pipeline Execution Dashboard/Log (High Priority)

- Real-time status for each phase: `[RUNNING] -> [COMPLETED] -> [METRICS]`
- Visual progress tracking that non-tech users can understand
- Key metrics displayed at each checkpoint:
  - Dataset size & label distribution
  - Training time & loss curves
  - Accuracy, F1, Precision, Recall per label
  - Sample predictions (correct & incorrect)

#### 2. Simplified Pipeline Flow

```
1. Data Generation → Track: sample count, label distribution, quality samples
2. Training → Track: loss, epochs, time, checkpoint path
3. Evaluation → Track: all metrics, error samples, confusion matrix
4. Decision Point → Track: improvement delta, decision rationale
5. [Optional] Continual Training → Track: same as Training
```

#### 3. Experiment Tracking Database Enhancement

- Each phase gets a comprehensive "report card"
- Store ALL intermediate outputs (predictions, error samples, etc.)
- Version everything (datasets, models, configs)

---

## Implementation Plan

### Step 1: Add Pipeline Observability Layer (1-2 days)

#### Create Pipeline Observer/Reporter Service

**New file**: `app/core/services/pipeline_observer.py`

```python
class PipelineObserver:
    """
    Tracks and reports pipeline progress in real-time
    - Logs each step with timestamps
    - Calculates and displays metrics
    - Generates human-readable summaries
    - Stores everything in DB for history
    """

    def start_phase(self, phase_name: str, phase_id: str) -> PhaseLog:
        """Initialize phase tracking"""
        pass

    def log_progress(self, phase_id: str, message: str, data: dict) -> None:
        """Log progress update with data"""
        pass

    def complete_phase(self, phase_id: str, summary: dict) -> PhaseReport:
        """Mark phase complete and generate summary"""
        pass

    def generate_comparison_report(self, current_phase: str, previous_phases: List[str]) -> ComparisonReport:
        """Compare current phase with previous phases"""
        pass
```

**Benefits**:
- Non-tech users see: "Training Phase: 75% complete, Current Loss: 0.23"
- Engineers see: Full logs, metrics, and data paths
- Everything is stored and retrievable

---

### Step 2: Simplify Pipeline API (Remove Error Buckets for Now)

#### New Simplified v2 Endpoints

```
POST /api/v2/workflow/pipeline/run
  - Runs complete cycle: Generate → Train → Evaluate → Report
  - Returns comprehensive status with metrics

GET /api/v2/workflow/pipeline/{pipeline_id}/status
  - Real-time status for running pipeline

GET /api/v2/workflow/pipeline/{pipeline_id}/phases/{phase_id}/report
  - Detailed report for specific phase

GET /api/v2/workflow/pipeline/{pipeline_id}/comparison
  - Compare all phases to see improvements
```

---

### Step 3: Enhanced Evaluation & Reporting

**Current issue**: Evaluation runs but results aren't easily digestible

**Proposed enhancement**: `app/core/services/evaluation_reporter.py`

```python
class EvaluationReporter:
    """
    Generate comprehensive, readable evaluation reports
    """

    def generate_report(self, eval_results: dict) -> Report:
        return {
            "summary": {
                "overall_accuracy": 0.85,
                "status": "GOOD" | "NEEDS_IMPROVEMENT" | "EXCELLENT",
                "recommendation": "Continue training" | "Stop - target reached" | "Investigate errors"
            },
            "per_label_breakdown": {
                "payment_intent": {
                    "f1": 0.87,
                    "precision": 0.89,
                    "recall": 0.85,
                    "support": 200,
                    "status": "GOOD",
                    "top_errors": [...]  # 5 most problematic samples
                },
                # ... other labels
            },
            "improvement_from_previous": {
                "accuracy_delta": +0.05,
                "f1_delta": +0.03,
                "interpretation": "Significant improvement"
            },
            "samples_visualization": {
                "correct_high_confidence": [...],  # 5 examples
                "correct_low_confidence": [...],   # 5 examples
                "incorrect_samples": [...]         # All errors with analysis
            },
            "confusion_matrix": {
                "matrix": [[...], [...], [...]],
                "labels": ["payment_intent", "payment_request", "open_intent"],
                "interpretation": "Most confusion between payment_intent and payment_request"
            }
        }
```

---

### Step 4: Add Quality Gates & Decision Points

**Problem**: Pipeline continues even if model is bad

**Solution**: Automatic quality checks - `app/core/services/pipeline_decision_maker.py`

```python
class PipelineDecisionMaker:
    """
    Decides whether to continue, stop, or retry based on metrics
    """

    def should_continue(self, current_metrics: dict, history: List[dict]) -> Decision:
        """
        Analyze metrics and decide next action

        Returns:
            Decision with action (CONTINUE, STOP, SUCCESS) and reason
        """

        # Check 1: Minimum quality threshold
        if current_metrics['f1_weighted'] < 0.6:
            return Decision(
                action="STOP",
                reason="Quality too low - F1 below 0.6 threshold",
                recommendation="Review training data quality and diversity"
            )

        # Check 2: Improvement check (last 3 phases)
        if len(history) > 3:
            recent_improvement = self._calculate_trend(history[-3:])
            if recent_improvement < 0.01:
                return Decision(
                    action="STOP",
                    reason="No significant improvement in last 3 phases",
                    recommendation="Try different approach: increase data, adjust hyperparameters"
                )

        # Check 3: Target reached
        if self._all_labels_above_threshold(current_metrics, target=0.85):
            return Decision(
                action="SUCCESS",
                reason="All labels achieved target F1 > 0.85",
                recommendation="Model ready for production"
            )

        # Check 4: Identify weak labels
        weak_labels = self._find_weak_labels(current_metrics, threshold=0.7)
        if weak_labels:
            return Decision(
                action="CONTINUE",
                reason=f"Labels need improvement: {', '.join(weak_labels)}",
                recommendation=f"Focus data generation on: {', '.join(weak_labels)}"
            )

        return Decision(
            action="CONTINUE",
            reason="Making progress, continue improving",
            recommendation="Continue with normal data generation"
        )

    def _calculate_trend(self, history: List[dict]) -> float:
        """Calculate improvement trend from history"""
        pass

    def _all_labels_above_threshold(self, metrics: dict, target: float) -> bool:
        """Check if all labels meet target"""
        pass

    def _find_weak_labels(self, metrics: dict, threshold: float) -> List[str]:
        """Identify labels below threshold"""
        pass
```

---

### Step 5: Implement Experiment Variation System (1 day)

**Purpose**: Test different configurations to find what works

**New file**: `app/core/schemas/experiment_config.py`

```python
from pydantic import BaseModel, Field

class ExperimentConfig(BaseModel):
    """Configuration for pipeline experiments"""

    # Data generation
    dataset_size_per_label: int = Field(200, description="Samples per label")
    use_seed_variations: bool = Field(True, description="Use multiple seed prompts")

    # Training
    training_epochs: int = Field(4, description="Number of training epochs")
    learning_rate: float = Field(1e-5, description="Learning rate")
    batch_size: int = Field(8, description="Training batch size")
    warmup_ratio: float = Field(0.15, description="Warmup ratio")

    # Model
    lora_r: int = Field(16, description="LoRA rank")
    lora_alpha: int = Field(32, description="LoRA alpha")

    # Quality gates
    min_f1_threshold: float = Field(0.6, description="Minimum acceptable F1")
    target_f1: float = Field(0.85, description="Target F1 for success")
    max_phases: int = Field(10, description="Max phases before stopping")

    def to_dict(self) -> dict:
        return self.model_dump()

    def get_experiment_name(self) -> str:
        """Generate descriptive name for this config"""
        return f"exp_data{self.dataset_size_per_label}_ep{self.training_epochs}_lr{self.learning_rate}"
```

**Add endpoint**:
```python
POST /api/v2/workflow/experiment/run
  Body: {
    "pipeline_id": "...",
    "config": ExperimentConfig,
    "description": "Testing with 500 samples per label"
  }
  - Run pipeline with specific config
  - Store results with config for comparison

GET /api/v2/workflow/experiments/compare
  Query: pipeline_id
  - Compare all experiments to find best config
  - Returns ranked list with metrics
```

---

### Step 6: Data Quality Insights (1 day)

**Problem**: Don't know if generated data is good quality

**Solution**: Add data quality checks - `app/core/services/data_quality_analyzer.py`

```python
from typing import List
from app.core.schemas import Sample

class DataQualityAnalyzer:
    """
    Analyze generated dataset quality
    """

    def analyze_dataset(self, samples: List[Sample]) -> QualityReport:
        """
        Comprehensive quality analysis of dataset

        Returns:
            QualityReport with scores and flags
        """
        return QualityReport(
            overall_score=0.85,
            diversity_score=self._calculate_diversity(samples),
            balance_score=self._calculate_balance(samples),
            length_distribution=self._analyze_lengths(samples),
            duplicate_rate=self._calculate_duplicate_rate(samples),
            quality_flags=self._identify_quality_issues(samples)
        )

    def _calculate_diversity(self, samples: List[Sample]) -> float:
        """
        Measure lexical diversity using:
        - Unique n-grams ratio
        - Vocabulary richness
        - Sentence structure variety
        """
        pass

    def _calculate_balance(self, samples: List[Sample]) -> float:
        """
        Check label distribution balance
        Perfect balance = 1.0
        """
        pass

    def _analyze_lengths(self, samples: List[Sample]) -> dict:
        """
        Return length statistics:
        - mean, median, min, max
        - distribution histogram
        """
        pass

    def _calculate_duplicate_rate(self, samples: List[Sample]) -> float:
        """Calculate percentage of near-duplicates"""
        pass

    def _identify_quality_issues(self, samples: List[Sample]) -> dict:
        """
        Flag potential issues:
        - too_short: samples < 5 words
        - too_similar: near-duplicates
        - potential_mislabels: contradictory samples
        """
        return {
            "too_short": [],
            "too_similar": [],
            "potential_mislabels": []
        }

class QualityReport(BaseModel):
    overall_score: float
    diversity_score: float
    balance_score: float
    length_distribution: dict
    duplicate_rate: float
    quality_flags: dict

    def get_summary(self) -> str:
        """Human-readable summary"""
        status = "EXCELLENT" if self.overall_score > 0.9 else \
                 "GOOD" if self.overall_score > 0.75 else \
                 "NEEDS_IMPROVEMENT"
        return f"Dataset Quality: {status} (Score: {self.overall_score:.2f})"
```

---

### Step 7: Simple Status Dashboard (Optional, 1-2 days)

For non-tech users, create a simple text-based view:

```
GET /api/v2/workflow/pipeline/{pipeline_id}/dashboard
```

**Returns**:
```
Pipeline: Payment Classification v2
Status: [████████░░] 80% - Training Phase

Current Phase: Training (Phase 3)
├─ Started: 2025-10-13 10:30:45
├─ Duration: 5m 23s
├─ Progress: Epoch 3/4
└─ Current Loss: 0.234

Latest Metrics:
├─ Accuracy: 85.3% (↑ +2.1% from Phase 2)
├─ F1 Score: 0.847 (↑ +0.031)
└─ Status: ✓ IMPROVING

Per Label Performance:
├─ payment_intent:   F1: 0.89 ✓
├─ payment_request:  F1: 0.83 ⚠️
└─ open_intent:      F1: 0.82 ⚠️

Next Steps: Evaluate model → Compare results → Decide
```

---

## Concrete Next Steps (Priority Order)

### Week 1: Foundation
1. ✅ **Create PipelineObserver service** - Tracks all pipeline steps
2. ✅ **Add progress logging to existing endpoints** - Make visible what's happening
3. ✅ **Create comprehensive evaluation reporter** - Better metrics presentation
4. ✅ **Add pipeline comparison endpoint** - Compare phases easily

### Week 2: Quality & Control
5. ✅ **Implement quality gates** - Automatic stop/continue decisions
6. ✅ **Add data quality analyzer** - Know if data is good
7. ✅ **Create experiment config system** - Test different settings
8. ✅ **Add simple status dashboard endpoint** - For non-tech users

### Week 3: Refinement & Experimentation
9. ✅ **Run baseline experiments** - Test with 100, 200, 500, 1000 samples per label
10. ✅ **Analyze and document findings** - What works, what doesn't
11. ✅ **Optimize based on results** - Update default configs
12. 🔄 **Once stable, re-add error bucket system** - But simpler version

---

## How This Helps

### For Non-Tech Users
- ✅ Clear status updates: "Training is 60% done, looking good!"
- ✅ Simple metrics: "Accuracy improved by 5% this round"
- ✅ Visual indicators: ✓ Good, ⚠️ Needs attention, ✗ Failed
- ✅ Understandable recommendations: "Focus on payment_request label"

### For Engineers
- ✅ Every experiment is tracked and comparable
- ✅ Easy to identify what configurations work
- ✅ Quick iteration: change config → run → compare
- ✅ Full data lineage: know exactly what generated each model
- ✅ Debugging friendly: all intermediate outputs saved

### For Pipeline Stability
- ✅ Quality gates prevent bad models from progressing
- ✅ Automatic stopping when no improvement
- ✅ Clear decision points with rationale
- ✅ All data is versioned and recoverable
- ✅ Experiment tracking prevents losing good configurations

---

## Immediate Actions (Start Here)

### 1. Create Test Pipeline Run Script

Create `scripts/test_pipeline.py`:
```python
"""
Quick test pipeline - completes in < 5 minutes
Perfect for testing pipeline changes
"""

config = ExperimentConfig(
    dataset_size_per_label=50,  # Small dataset
    training_epochs=2,          # Fast training
    # ... other settings
)

# Run pipeline with observer
observer = PipelineObserver()
# ... run and track
```

### 2. Add Comprehensive Logging

Update existing endpoints in `app/api/routes/v2/workflow.py`:
- Add observer.start_phase() at start
- Add observer.log_progress() during execution
- Add observer.complete_phase() at end
- Store everything in database

### 3. Create Comparison View

New endpoint:
```python
@router.get("/pipeline/{pipeline_id}/phases/compare")
async def compare_phases(pipeline_id: str, phase_ids: List[str]):
    """
    Compare multiple phases side-by-side
    Shows: metrics, data sizes, improvements, time taken
    """
    pass
```

---

## Key Metrics to Track

### Data Generation Phase
- Samples generated per label
- Generation time
- Duplicate rate
- Quality score
- Storage path

### Training Phase
- Training time
- Final loss
- Epochs completed
- Checkpoint path
- Model size

### Evaluation Phase
- Overall accuracy, precision, recall, F1
- Per-label metrics (all 4 for each label)
- Confusion matrix
- Error sample count
- Low confidence sample count

### Phase Comparison
- Improvement deltas (vs previous phase)
- Time trend (getting faster/slower?)
- Quality trend (improving/plateauing?)
- Decision recommendation

---

## Database Schema Additions

### New Table: `phase_logs`
```sql
CREATE TABLE phase_logs (
    id TEXT PRIMARY KEY,
    phase_id TEXT REFERENCES pipeline_phase(id),
    timestamp TIMESTAMP,
    log_level TEXT,  -- INFO, WARNING, ERROR
    message TEXT,
    data TEXT,  -- JSON data
    created_at TIMESTAMP
);
```

### New Table: `experiment_runs`
```sql
CREATE TABLE experiment_runs (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT REFERENCES pipeline(id),
    config TEXT,  -- JSON ExperimentConfig
    description TEXT,
    status TEXT,  -- running, completed, failed
    phases TEXT,  -- JSON list of phase IDs
    best_f1 FLOAT,
    best_phase_id TEXT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### New Table: `quality_reports`
```sql
CREATE TABLE quality_reports (
    id TEXT PRIMARY KEY,
    phase_id TEXT REFERENCES pipeline_phase(id),
    dataset_file_id TEXT REFERENCES dataset_file(id),
    overall_score FLOAT,
    diversity_score FLOAT,
    balance_score FLOAT,
    duplicate_rate FLOAT,
    quality_flags TEXT,  -- JSON
    created_at TIMESTAMP
);
```

---

## Testing Strategy

### Unit Tests
- Test each service independently
- Mock external dependencies
- Test decision logic with various scenarios

### Integration Tests
- Test complete pipeline flow
- Verify database persistence
- Check report generation

### End-to-End Tests
- Run actual mini pipeline (50 samples, 2 epochs)
- Verify all steps complete
- Check all data is stored correctly

---

## Success Criteria

After implementation, we should be able to:

1. ✅ Run pipeline and see real-time progress
2. ✅ Know immediately if data quality is good
3. ✅ Get automatic recommendations (continue/stop/investigate)
4. ✅ Compare experiments easily to find best config
5. ✅ Explain to non-tech users what's happening
6. ✅ Debug issues quickly with full logs
7. ✅ Reproduce any result with stored configs
8. ✅ Identify optimal dataset size through experiments

---

## Future Enhancements (After Stabilization)

1. **Re-introduce Error Buckets** (Simplified)
   - Automatic error categorization
   - Targeted data generation for weak areas
   - Track improvement per error category

2. **A/B Testing**
   - Run multiple configs in parallel
   - Automatic best model selection

3. **Model Registry**
   - Tag models (dev, staging, production)
   - Rollback capability

4. **Cost Tracking**
   - Track LLM API costs per phase
   - Optimize for cost/performance

5. **Web Dashboard**
   - Visual charts and graphs
   - Interactive exploration
   - Export reports

---

## Questions to Answer Through Experiments

1. **Dataset Size**: How many samples per label are optimal?
   - Test: 100, 200, 500, 1000 samples

2. **Data Diversity**: Does more diverse data help?
   - Test: Single seed prompt vs multiple variants

3. **Training Duration**: More epochs = better model?
   - Test: 2, 4, 6, 10 epochs

4. **Continual Training**: When does it help vs hurt?
   - Test: Fresh training vs continual training

5. **Learning Rate**: Optimal learning rate?
   - Test: 1e-4, 5e-5, 1e-5, 5e-6

Track all experiments and analyze results to find patterns.

---

## Implementation Priority

**Phase 1 (This Week)**:
- [ ] PipelineObserver service
- [ ] EvaluationReporter service
- [ ] Add logging to existing endpoints
- [ ] Create comparison endpoint

**Phase 2 (Next Week)**:
- [ ] PipelineDecisionMaker
- [ ] DataQualityAnalyzer
- [ ] ExperimentConfig system
- [ ] Test pipeline script

**Phase 3 (Week 3)**:
- [ ] Run experiments (various configs)
- [ ] Document findings
- [ ] Optimize defaults
- [ ] Create status dashboard

**Phase 4 (Future)**:
- [ ] Re-add error buckets (simplified)
- [ ] Web dashboard
- [ ] Advanced features

---

## Contact & Feedback

Track progress and issues in this document. Update status as components are completed.

**Last Updated**: 2025-10-13
