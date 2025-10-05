from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from .workflow import Sample

class Prediction(BaseModel):
    label: str
    prob: float
    dis: Optional[float] = None  # Distance for ADB
    closest: Optional[str] = None  # For unknown predictions

class TestCase(BaseModel):
    input: Sample
    true_label: str
    prediction: Optional[Prediction] = None

class LabelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

class OverallMetrics(BaseModel):
    accuracy: float
    unknown_rate: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_samples: int
    correct_predictions: int
    unknown_predictions: int

class MisclassifiedOpenIntent(BaseModel):
    text: str
    predicted_as: str
    confidence: float
    distance: float

class OpenIntentAnalysis(BaseModel):
    total_samples: int
    detected_as_unknown: int
    unknown_rate: float
    false_positive_rate: float
    misclassified: List[MisclassifiedOpenIntent]

class ErrorCase(BaseModel):
    """Represents a single error case with input and predictions"""
    input: Sample
    true_label: str
    predicted_label: str
    confidence: float
    distance: Optional[float] = None

class ErrorsByLabel(BaseModel):
    """Groups error cases by label"""
    false_positives: List[ErrorCase]  # Cases predicted as this label but shouldn't be
    false_negatives: List[ErrorCase]  # Cases that should be this label but weren't predicted as such


class EvaluationResult(BaseModel):
    overall: OverallMetrics
    per_label: Dict[str, LabelMetrics]
    adb_info: Optional[Dict[str, Any]] = None
    test_cases: Optional[List[TestCase]] = None
    open_intent_analysis: Optional[OpenIntentAnalysis] = None
    unknown_analysis: Optional[Dict[str, Any]] = None
    errors_by_label: Optional[Dict[str, ErrorsByLabel]] = None  # Grouped FP/FN by label

class DataGenerationAction(BaseModel):
    """Specific data generation action that can be programmatically executed"""
    label: str = Field(description="The label for which to generate/modify examples")
    # expected_count: int = Field(description="Expected number of samples to generate/modify")

    # Specific instructions
    keywords_to_include: List[str] = Field(default=[], description="Specific keywords/phrases that should appear in generated examples")
    keywords_to_avoid: List[str] = Field(default=[], description="Keywords/phrases to avoid or filter out")
    # word_replacements: Dict[str, List[str]] = Field(default={}, description="Word replacement mappings: original -> [alternatives]")

    # Pattern specifications
    sentence_patterns: List[str] = Field(default=[], description="Specific sentence structures/patterns to follow")
    context_requirements: List[str] = Field(default=[], description="Required contextual elements")
    diversity_constraints: List[str] = Field(default=[], description="Diversity requirements (e.g., 'vary sentence length', 'use different personas')")
    ignore: bool = Field(default=False, description="True if no action needed for this label")

    # Distribution targets
    # min_examples: Optional[int] = Field(default=None, description="Minimum examples needed for this pattern")
    # target_distribution: Optional[float] = Field(default=None, description="Target percentage of total dataset")

class ErrorPatternAnalysis(BaseModel):
    """LLM-generated analysis of error patterns for misclassified examples"""
    predicted_label: str = Field(description="The incorrect label that the model predicted")
    expected_label: str = Field(description="The correct label that should have been predicted")

    # What the LLM identified
    identified_issues: List[str] = Field(
        description="Specific, concrete issues found in the data (e.g., 'Missing examples with pronoun I as recipient', 'Overuse of word pay in payment_request')"
    )

    # Programmable fixes
    data_actions: List[DataGenerationAction] = Field(
        description="Specific, programmable actions to fix the identified issues"
    )


class BuildPromptRequest(BaseModel):
    actions: List[DataGenerationAction]