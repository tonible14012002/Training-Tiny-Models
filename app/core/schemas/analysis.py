from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Any
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
    coverage: float
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
    coverage: float
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

class EvaluationResult(BaseModel):
    overall: OverallMetrics
    per_label: Dict[str, LabelMetrics]
    adb_info: Optional[Dict[str, Any]] = None
    test_cases: Optional[List[TestCase]] = None
    open_intent_analysis: Optional[OpenIntentAnalysis] = None


class ErrorBucket(BaseModel):
    name: str
    description: Optional[str] = None
    approach: str  # The strategy to overcome this error
    example_issue: Optional[str] = None  # Example of the error manifestation
    data_generation_strategy: Optional[str] = None  # Specific data generation approach

EXAMPLE_ERROR_BUCKETS = [
    ErrorBucket(
        name="Over reliance on keywords",
        description="The model relies too heavily on specific keywords and misses the overall context of the message.",
        approach="Generate diverse examples with varied vocabulary and synonyms to reduce keyword dependency",
        example_issue="Model classifies 'transfer money' as payment_intent but misses 'send funds' with same intent",
        data_generation_strategy="Create examples using synonyms, paraphrasing, and different linguistic expressions for same intent"
    ),
    ErrorBucket(
        name="Over focusing on unrelated word",
        description="The model attends to an unrelated word that might not related to the actual intent.",
        approach="Generate examples with distracting words to improve context understanding",
        example_issue="Model focuses on 'payment' in 'payment reminder' and classifies as payment_intent instead of payment_request",
        data_generation_strategy="Include examples with potentially confusing words in different contexts"
    ),
    ErrorBucket(
        name="Miss focus on important word",
        description="The model fails to attend to an important word that is crucial for determining the intent.",
        approach="Generate examples that emphasize critical intent-determining words in various contexts",
        example_issue="Model misses 'request' in 'I request payment' and classifies as payment_intent",
        data_generation_strategy="Create examples highlighting key intent words in different sentence positions"
    ),
    ErrorBucket(
        name="Ambiguous intent",
        description="The input message is ambiguous and could reasonably be interpreted as having multiple intents.",
        approach="Generate clearer, more explicit examples to establish stronger intent boundaries",
        example_issue="'Can you handle the payment?' could be payment_request or smart_payment_system_command",
        data_generation_strategy="Create unambiguous examples with clear intent signals and context"
    ),
    ErrorBucket(
        name="Complex sentence structure",
        description="The input message has a complex sentence structure that makes it difficult to parse the intent.",
        approach="Generate examples with varied sentence complexities to improve structural understanding",
        example_issue="Model struggles with 'If possible, could you maybe send me the payment when convenient?'",
        data_generation_strategy="Include examples with complex grammar, nested clauses, and varied sentence structures"
    ),
    ErrorBucket(
        name="Miss detecting simple intent",
        description="The model fails to detect a simple intent that is clearly expressed in the input message.",
        approach="Generate more simple, everyday conversational examples to improve basic intent recognition",
        example_issue="Model misclassifies simple 'pay me' as smart_payment_system_command instead of payment_request",
        data_generation_strategy="Create straightforward, colloquial examples using common everyday language"
    ),
]