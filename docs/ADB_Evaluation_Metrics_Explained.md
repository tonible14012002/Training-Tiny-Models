# ADB Evaluation Metrics Explained

This document explains the comprehensive evaluation metrics used in the `evaluate` function from `/src/payment_classifier/inference/adb_inference.py`. The Angular Distance-Based (ADB) evaluation system provides detailed metrics for intent classification performance.

## Overview

The `evaluate` function evaluates model performance using Angular Distance-Based classification on datasets with known ground truth labels. It treats classification as multiple binary problems (one-vs-all) and provides both overall and per-label metrics.

> **Important Note**: This evaluation is designed for KNOWN intents only. All samples are expected to be valid known intents and should NOT be predicted as "Unknown". The "Unknown" prediction is treated as a classification error.

## Overall Metrics

### Core Performance Metrics

#### 1. **Accuracy** (`accuracy`)
- **Definition**: Proportion of samples where the predicted label exactly matches the true label
- **Formula**: `correct_predictions / total_samples`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Overall classification performance across all labels

#### 2. **Coverage** (`coverage`)
- **Definition**: Proportion of samples that were NOT predicted as "Unknown"
- **Formula**: `(total_samples - unknown_predictions) / total_samples`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: How well the model provides confident predictions rather than rejecting samples as unknown

#### 3. **Unknown Rate** (`unknown_rate`)
- **Definition**: Proportion of samples predicted as "Unknown"
- **Formula**: `unknown_predictions / total_samples`
- **Range**: 0.0 to 1.0 (lower is better for known intent evaluation)
- **Interpretation**: Rate at which the model fails to classify known intents

#### 4. **Known Intent True Negative Rate** (`known_intent_true_negative_rate`)
- **Definition**: Proportion of known intents correctly NOT classified as "Unknown"
- **Formula**: `(total_samples - unknown_predictions) / total_samples`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Model's ability to recognize that samples belong to known intent classes

### Macro-Averaged Metrics

#### 5. **Macro Precision** (`macro_precision`)
- **Definition**: Average precision across all labels that exist in the dataset
- **Formula**: `sum(precision_per_label) / number_of_valid_labels`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Average ability to avoid false positives across all intent classes

#### 6. **Macro Recall** (`macro_recall`)
- **Definition**: Average recall across all labels that exist in the dataset
- **Formula**: `sum(recall_per_label) / number_of_valid_labels`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Average ability to find all positive instances across all intent classes

#### 7. **Macro F1** (`macro_f1`)
- **Definition**: Average F1-score across all labels that exist in the dataset
- **Formula**: `sum(f1_per_label) / number_of_valid_labels`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Balanced average performance across all intent classes

## Per-Label Metrics

Each intent label (including "Unknown") gets individual binary classification metrics:

### Binary Classification Metrics

#### 1. **Precision** (`precision`)
- **Definition**: Of all samples predicted as this label, how many were actually this label
- **Formula**: `true_positives / (true_positives + false_positives)`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Quality of positive predictions for this label

#### 2. **Recall** (`recall`)
- **Definition**: Of all actual samples of this label, how many were correctly predicted
- **Formula**: `true_positives / (true_positives + false_negatives)`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Completeness of detection for this label

#### 3. **F1-Score** (`f1_score`)
- **Definition**: Harmonic mean of precision and recall
- **Formula**: `2 * (precision * recall) / (precision + recall)`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Balanced measure of precision and recall

#### 4. **Label Accuracy** (`accuracy`)
- **Definition**: Binary classification accuracy for this specific label
- **Formula**: `(true_positives + true_negatives) / total_samples`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Overall correctness for this label's binary classification

#### 5. **Label Coverage** (`coverage`)
- **Definition**: For known labels: proportion of this label's samples that were NOT predicted as "Unknown"
- **Formula**: `(total_samples_of_label - unknown_predictions_of_label) / total_samples_of_label`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: How well the model confidently classifies samples of this specific intent

### Confusion Matrix Components

#### 6. **True Positives** (`true_positives`)
- **Definition**: Samples correctly predicted as this label
- **Interpretation**: Successful detections of this intent

#### 7. **False Positives** (`false_positives`)
- **Definition**: Samples incorrectly predicted as this label
- **Interpretation**: Over-classification errors for this intent

#### 8. **True Negatives** (`true_negatives`)
- **Definition**: Samples correctly predicted as NOT this label
- **Interpretation**: Successful rejections of this intent

#### 9. **False Negatives** (`false_negatives`)
- **Definition**: Samples of this label incorrectly predicted as something else
- **Interpretation**: Missed detections of this intent

#### 10. **Samples** (`samples`)
- **Definition**: Total number of actual samples of this label in the dataset
- **Formula**: `true_positives + false_negatives`
- **Interpretation**: Ground truth count for this intent class

## Unknown Predictions Breakdown

### Per-Label Unknown Analysis

For each true label, the system tracks:

#### 1. **Unknown Predictions** (`unknown_predictions`)
- **Definition**: Number of samples of this true label that were predicted as "Unknown"
- **Interpretation**: How many samples of this intent were rejected by the model

#### 2. **Total Samples** (`total_samples`)
- **Definition**: Total number of samples of this true label in the dataset
- **Interpretation**: Ground truth count for analysis

#### 3. **Unknown Rate** (`unknown_rate`)
- **Definition**: Proportion of this label's samples predicted as "Unknown"
- **Formula**: `unknown_predictions / total_samples`
- **Range**: 0.0 to 1.0 (lower is better)
- **Interpretation**: Rejection rate for this specific intent

## ADB Information

### Technical Details

#### 1. **Radii** (`radii`)
- **Definition**: Distance thresholds for each intent class in embedding space
- **Usage**: Determines the boundary for classifying samples as belonging to each intent
- **Interpretation**: Larger radii = more inclusive classification for that intent

#### 2. **Labels** (`labels`)
- **Definition**: Mapping from label IDs to label names
- **Usage**: Reference for understanding the intent classes being evaluated
- **Interpretation**: The complete set of known intent classes in the model

## Evaluation Context

### Key Assumptions
1. **Known Intents Only**: All evaluation samples are expected to be valid known intents
2. **Binary Treatment**: Each label is treated as a separate binary classification problem
3. **Unknown as Error**: "Unknown" predictions are considered classification failures for known intents
4. **Coverage Focus**: Emphasis on the model's ability to provide confident classifications

### Use Cases
- **Model Performance Assessment**: Comprehensive view of classification quality
- **Intent Coverage Analysis**: Understanding which intents are well-covered vs. frequently rejected
- **Error Pattern Analysis**: Identifying systematic classification issues
- **Threshold Tuning**: Evaluating the impact of ADB radius settings

### Limitations
- Does not evaluate out-of-domain/unknown intent detection capability
- Assumes all evaluation samples should be classifiable as known intents
- May not reflect real-world performance where unknown intents are common

For testing unknown intent detection, use `evaluate_with_unknown_intents()` instead.