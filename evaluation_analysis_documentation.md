# Model Evaluation and Analysis Documentation

This document provides a comprehensive overview of the evaluation and analysis system used in the iterative LLM fine-tuning framework, with detailed explanations of each metric and implementation.

## Overview

The evaluation system consists of two main components:
1. **ADBModelInference** (`src/payment_classifier/inference/adb_inference.py`): Handles ADB (Adaptive Decision Boundary) based inference and evaluation
2. **ModelAnalyzer** (`app/core/services/model_analyzer/model_analyzer.py`): Provides comprehensive model analysis and error categorization

## ADB (Adaptive Decision Boundary) Methodology

### What is ADB?

ADB is a novel approach for intent classification that creates adaptive decision boundaries around each intent class in the embedding space. Instead of relying solely on softmax probabilities, ADB:

1. **Calculates intent centers**: Mean embeddings for each intent class from training data
2. **Determines optimal radii**: Uses percentile-based approach (90th percentile by default) to set decision boundaries
3. **Makes predictions**: Classifies samples based on their distance to intent centers relative to the learned radii

### ADB Implementation Details

#### Center Calculation (`calc_adb` method - lines 97-155)
```python
# Calculate centers as mean of class embeddings
intent_centers[label] = label_sums[label] / label_counts[label]

# Calculate radius using 90th percentile of distances
radius = torch.quantile(distances, confidence_ratio)  # confidence_ratio = 0.9
```

#### Prediction Logic (`predict_with_adb` method - lines 157-212)
- **Within Radius**: If embedding falls within the radius of an intent center, it's classified as that intent
- **Multiple Intents**: If within multiple radii, chooses the one with highest softmax probability
- **Unknown Intent**: If outside all radii, classified as "Unknown"

## Evaluation Metrics

### Overall Metrics

#### 1. Accuracy
**Definition**: Percentage of correct predictions among all samples
**Calculation**: `correct_predictions / total_samples`
**Range**: [0, 1]
**Interpretation**: Higher is better. Measures overall model performance on known intents.

#### 2. Coverage
**Definition**: Percentage of samples that receive a classification (not "Unknown")
**Calculation**: `(total_samples - unknown_predictions) / total_samples`
**Range**: [0, 1]
**Interpretation**: Higher coverage means the model is confident in more predictions. Trade-off with precision.

#### 3. Unknown Rate
**Definition**: Percentage of samples classified as "Unknown"
**Calculation**: `unknown_predictions / total_samples`
**Range**: [0, 1]
**Interpretation**: Lower is better for known intent evaluation. Complement of coverage.

#### 4. Macro Precision
**Definition**: Average of per-class precision scores
**Calculation**: `sum(precision_per_class) / number_of_classes`
**Range**: [0, 1]
**Interpretation**: Measures average precision across all classes, giving equal weight to each class regardless of support.

#### 5. Macro Recall
**Definition**: Average of per-class recall scores
**Calculation**: `sum(recall_per_class) / number_of_classes`
**Range**: [0, 1]
**Interpretation**: Measures average recall across all classes, giving equal weight to each class.

#### 6. Macro F1-Score
**Definition**: Average of per-class F1 scores
**Calculation**: `sum(f1_per_class) / number_of_classes`
**Range**: [0, 1]
**Interpretation**: Harmonic mean of macro precision and recall. Balanced measure of model performance.

### Per-Label Metrics

For each intent class, the following metrics are calculated:

#### 1. Label Accuracy
**Definition**: Percentage of correct predictions for samples of this specific label
**Calculation**: `correct_for_label / total_samples_for_label`
**Implementation**: Lines 293-294 in `adb_inference.py`

#### 2. Label Coverage
**Definition**: Percentage of samples of this label that received a classification
**Calculation**: `(total_for_label - unknown_for_label) / total_for_label`
**Implementation**: Lines 294-295 in `adb_inference.py`

#### 3. Precision
**Definition**: Among predictions made for this label, percentage that were correct
**Calculation**: `true_positives / (true_positives + false_positives)`
**Implementation**: Lines 288 in `adb_inference.py`

#### 4. Recall
**Definition**: Among actual samples of this label, percentage that were correctly identified
**Calculation**: `true_positives / (true_positives + false_negatives)`
**Implementation**: Lines 289 in `adb_inference.py`

#### 5. F1-Score
**Definition**: Harmonic mean of precision and recall
**Calculation**: `2 * (precision * recall) / (precision + recall)`
**Implementation**: Lines 290 in `adb_inference.py`

#### 6. Support Metrics
- **Samples**: Total number of samples for this label
- **True Positives**: Correctly predicted as this label
- **False Positives**: Incorrectly predicted as this label
- **False Negatives**: This label incorrectly predicted as something else

## Open Intent Analysis

### Purpose
Evaluates the model's ability to detect out-of-distribution (OOD) samples that don't belong to any trained intent class.

### Key Metrics

#### 1. Unknown Detection Rate
**Definition**: Percentage of open intent samples correctly identified as "Unknown"
**Calculation**: `detected_as_unknown / total_open_intent_samples`
**Implementation**: Lines 153-155 in `model_analyzer.py`

#### 2. False Positive Rate
**Definition**: Percentage of open intent samples incorrectly classified as known intents
**Calculation**: `(total_samples - detected_as_unknown) / total_samples`
**Implementation**: Lines 173 in `model_analyzer.py`

#### 3. Misclassified Analysis
Detailed breakdown of open intent samples that were incorrectly classified:
- **Text**: The misclassified sample
- **Predicted As**: Which known intent it was classified as
- **Confidence**: Model's confidence in the wrong prediction
- **Distance**: Embedding distance to the predicted intent center

## Error Analysis System

### Error Categorization

The system automatically categorizes prediction errors into predefined buckets to guide data generation:

#### 1. Over Reliance on Keywords
**Description**: Model depends too heavily on specific words
**Example**: Classifies "transfer money" correctly but misses "send funds"
**Data Strategy**: Generate diverse vocabulary and synonyms

#### 2. Over Focusing on Unrelated Words
**Description**: Model attends to irrelevant words
**Example**: Focuses on "payment" in "payment reminder" and misclassifies
**Data Strategy**: Include distracting words in different contexts

#### 3. Miss Focus on Important Words
**Description**: Model ignores crucial intent-determining words
**Example**: Misses "request" in "I request payment"
**Data Strategy**: Emphasize key words in various sentence positions

#### 4. Ambiguous Intent
**Description**: Input could reasonably have multiple intents
**Example**: "Can you handle the payment?" (request vs. command)
**Data Strategy**: Create clearer, more explicit examples

#### 5. Complex Sentence Structure
**Description**: Complex grammar confuses the model
**Example**: "If possible, could you maybe send me the payment when convenient?"
**Data Strategy**: Include varied sentence complexities

#### 6. Miss Detecting Simple Intent
**Description**: Fails on straightforward expressions
**Example**: Misclassifies simple "pay me" as wrong intent
**Data Strategy**: Generate everyday conversational examples

### Error Detection Logic

The system uses heuristics to categorize errors (lines 198-238 in `model_analyzer.py`):
- **Message length**: Short vs. long sentences
- **Keyword presence**: Payment-related terms
- **Structural complexity**: Commas, conditional words
- **Prediction confidence**: High confidence wrong answers indicate keyword reliance

## Analysis Workflow

### 1. Model Loading
```python
analyzer.load_model(checkpoint_path)  # Loads ADB model with centers and radii
```

### 2. Comprehensive Analysis
```python
result = analyzer.analyze_model(
    evaluation_dataset=eval_data,
    open_intent_samples=open_samples,
    include_test_cases=True
)
```

### 3. Error Analysis
```python
error_buckets = analyzer.analyze_errors(result)
```

### 4. Report Generation
```python
report = analyzer.generate_analysis_report(result)
```

## Data Structures

### EvaluationResult Schema
- **overall**: OverallMetrics with system-wide performance
- **per_label**: Dict of LabelMetrics for each intent class
- **adb_info**: ADB configuration (radii and label mappings)
- **test_cases**: Optional individual sample results
- **open_intent_analysis**: Optional OOD detection results

## Key Implementation Files

1. **ADBModelInference** (`src/payment_classifier/inference/adb_inference.py:214-331`): Core evaluation logic
2. **ModelAnalyzer** (`app/core/services/model_analyzer/model_analyzer.py:41-107`): Orchestration and analysis
3. **Analysis Schemas** (`app/core/schemas/analysis.py`): Data structures and error buckets

## Usage in the Training Loop

The evaluation system integrates with the iterative training process:

1. **After Training**: Model checkpoints include ADB data
2. **Evaluation**: Comprehensive analysis on held-out test set
3. **Error Analysis**: Identifies specific model weaknesses
4. **Data Generation**: Uses error buckets to guide synthetic data creation
5. **Iteration**: Process repeats with improved training data

This systematic approach enables targeted improvement of model performance through data-driven insights.