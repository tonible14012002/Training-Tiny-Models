# ErrorReasoner Service Documentation

## Overview

The `ErrorReasoner` service is a critical component in the iterative fine-tuning system that bridges the gap between error detection and targeted data generation. It analyzes misclassification patterns to provide detailed reasoning about why specific errors occurred, which then guides strategic synthetic data generation.

## Architecture

```
ModelAnalyzer → ErrorReasoner → DataGenerator
     ↓              ↓              ↓
Error Detection → Root Cause → Targeted Data
                 Analysis      Generation
```

## Location
- **Main Class**: `app/core/services/error_reasoner/error_reasoner.py`
- **Module**: `app/core/services/error_reasoner/__init__.py`
- **Integration**: `app/core/services/__init__.py`

## Core Data Structures

### ErrorReason
Represents a specific reason why a misclassification occurred:
```python
@dataclass
class ErrorReason:
    reason_type: str              # Category of error (e.g., "keyword_confusion")
    description: str              # Human-readable explanation
    confidence: float             # Confidence in this reason (0-1)
    evidence: Dict               # Supporting evidence and metrics
    data_generation_prompt: str  # Prompt fragment for LLM data generation
```

### MisclassificationAnalysis
Analysis results for a single misclassified test case:
```python
@dataclass
class MisclassificationAnalysis:
    test_case: TestCase
    primary_reasons: List[ErrorReason]    # Most likely reasons
    secondary_reasons: List[ErrorReason]  # Possible but less likely reasons
    overall_confidence: float             # Overall confidence in analysis
```

## Current Implementation Status

### ✅ Completed (Basic Structure)

#### 1. Core Class Framework
- Basic `ErrorReasoner` class structure
- Data structures for error analysis
- Integration with existing services
- Reason templates initialization

#### 2. Basic Analysis Methods
- `analyze_misclassifications()`: Main entry point for error analysis
- `_analyze_single_case()`: Individual test case analysis with basic heuristics
- `generate_data_generation_prompts()`: Convert error reasons to generation prompts
- `get_error_summary()`: Statistical summary of error patterns

#### 3. Reason Templates
Pre-defined templates for common error types:
- **keyword_confusion**: Model confused by keywords in wrong context
- **missing_context**: Lacks contextual understanding
- **structural_complexity**: Complex sentence structure issues
- **semantic_similarity**: Confusion between similar intents
- **ambiguous_phrasing**: Ambiguous input interpretation

### 🚧 Placeholder Methods (To Be Implemented)

#### 1. Advanced Text Analysis
```python
def _analyze_keyword_conflicts(self, test_case: TestCase) -> List[ErrorReason]:
    # TODO: Extract key terms and identify conflicts
    # TODO: Check for terms associated with different intents
    # TODO: Analyze semantic similarity between conflicting terms
```

#### 2. Context Analysis
```python
def _analyze_context_adequacy(self, test_case: TestCase) -> List[ErrorReason]:
    # TODO: Check for context clues in text
    # TODO: Identify missing information for intent clarity
    # TODO: Assess sentence completeness
```

#### 3. Semantic Similarity Analysis
```python
def _analyze_semantic_similarity(self, test_case: TestCase) -> List[ErrorReason]:
    # TODO: Compare embedding similarity between intent classes
    # TODO: Identify potential confusion boundaries
    # TODO: Suggest disambiguation strategies
```

#### 4. Structural Pattern Analysis
```python
def _analyze_structural_patterns(self, test_case: TestCase) -> List[ErrorReason]:
    # TODO: Parse sentence structure
    # TODO: Identify complex grammatical constructions
    # TODO: Assess impact on intent recognition
```

## Current Basic Analysis Logic

The current implementation uses simple heuristics:

### High Confidence Misclassification
```python
if confidence > 0.8:
    # Suggests keyword confusion - model is confident but wrong
    reason = ErrorReason(
        reason_type="keyword_confusion",
        description=f"High confidence misclassification suggests keyword confusion",
        confidence=0.7,
        data_generation_prompt=f"Generate examples that distinguish {true_label} from {predicted_label}"
    )
```

### Structural Complexity Detection
```python
if len(text.split()) > 15:
    # Long sentences suggest structural complexity issues
    reason = ErrorReason(
        reason_type="structural_complexity",
        description="Complex sentence structure may have hindered classification",
        confidence=0.6,
        data_generation_prompt=f"Create examples with varied sentence structures"
    )
```

## Integration Workflow

### 1. Input Processing
```python
# From ModelAnalyzer
evaluation_result = analyzer.analyze_model(eval_data, include_test_cases=True)
error_buckets = analyzer.analyze_errors(evaluation_result)

# To ErrorReasoner
reasoner = ErrorReasoner()
analyses = reasoner.analyze_misclassifications(evaluation_result, error_buckets)
```

### 2. Error Analysis
```python
# Generate detailed analysis for each misclassified case
for test_case in misclassified_cases:
    analysis = reasoner._analyze_single_case(test_case, error_buckets)
    # Returns MisclassificationAnalysis with primary/secondary reasons
```

### 3. Prompt Generation
```python
# Convert error reasons into actionable data generation prompts
prompts = reasoner.generate_data_generation_prompts(analyses, max_prompts=10)
# Returns list of targeted prompts for DataGenerator
```

### 4. Summary Statistics
```python
# Get overview of error patterns
summary = reasoner.get_error_summary(analyses)
# Returns: total_cases, reason_distribution, average_confidence, insights
```

## Future Development Roadmap

### Phase 1: Enhanced Text Analysis
- **Keyword Extraction**: Use NLP libraries (spaCy, NLTK) to extract key terms
- **Conflict Detection**: Identify terms that appear in multiple intent contexts
- **Semantic Analysis**: Use word embeddings to detect semantic conflicts

### Phase 2: Context Understanding
- **Context Clue Detection**: Identify implicit context in text
- **Information Gap Analysis**: Determine missing information that would clarify intent
- **Sentence Completeness**: Assess grammatical and semantic completeness

### Phase 3: Advanced Reasoning
- **Intent Boundary Analysis**: Use model embeddings to understand confusion boundaries
- **Pattern Recognition**: Machine learning-based error pattern detection
- **Confidence Calibration**: Improve confidence scoring for error reasons

### Phase 4: LLM Integration
- **LLM-Powered Analysis**: Use teacher LLM to analyze error reasons
- **Dynamic Prompt Generation**: Generate context-aware prompts for data generation
- **Reasoning Validation**: Use LLM to validate and improve error reasoning

## Usage Examples

### Basic Usage
```python
from app.core.services import ErrorReasoner, ModelAnalyzer

# Initialize services
analyzer = ModelAnalyzer(trainer_service, data_manager)
reasoner = ErrorReasoner()

# Load model and analyze
analyzer.load_model(checkpoint_path)
eval_result = analyzer.analyze_model(eval_data, include_test_cases=True)

# Analyze errors
analyses = reasoner.analyze_misclassifications(eval_result)

# Generate targeted prompts
prompts = reasoner.generate_data_generation_prompts(analyses)

# Get summary
summary = reasoner.get_error_summary(analyses)
```

### Integration with Training Loop
```python
# After model evaluation
if eval_result.overall.accuracy < target_accuracy:
    # Analyze specific error patterns
    error_analyses = reasoner.analyze_misclassifications(eval_result)

    # Generate targeted data generation prompts
    generation_prompts = reasoner.generate_data_generation_prompts(error_analyses)

    # Use prompts to guide next data generation iteration
    for prompt in generation_prompts:
        data_generator.generate_targeted_data(prompt, num_samples=50)
```

## Key Benefits

1. **Targeted Improvement**: Identifies specific model weaknesses for focused data generation
2. **Automated Analysis**: Reduces manual error analysis effort
3. **Scalable Reasoning**: Can analyze hundreds of misclassifications systematically
4. **Strategic Data Generation**: Converts error insights into actionable generation strategies
5. **Continuous Learning**: Error patterns inform iterative model improvement

## Dependencies

### Current
- `app.core.schemas.analysis`: TestCase, ErrorBucket, EvaluationResult
- `typing`, `dataclasses`: Core Python typing support
- `logging`: Standard logging functionality

### Future (Planned)
- `spacy` or `nltk`: Advanced text analysis
- `transformers`: Embedding-based semantic analysis
- `sklearn`: Pattern recognition and clustering
- Teacher LLM integration: Dynamic reasoning and prompt generation

## Testing Strategy

### Unit Tests (Planned)
- Test individual error reason generation
- Validate prompt generation logic
- Test summary statistics calculation

### Integration Tests (Planned)
- End-to-end workflow with real misclassification data
- Integration with ModelAnalyzer and DataGenerator
- Performance testing with large error datasets

### Evaluation Metrics (Planned)
- **Reasoning Accuracy**: Manual validation of error reason correctness
- **Prompt Effectiveness**: Measure improvement in generated data quality
- **Coverage**: Percentage of errors that receive meaningful analysis

This service is designed to be the intelligent bridge between error detection and targeted improvement, enabling the system to learn from its mistakes and generate precisely the data needed to address specific weaknesses.