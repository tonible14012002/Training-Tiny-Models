from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
from .workflow import Sample

class Prediction(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    label_probabilities: Dict[str, float]
    original_sample: Optional[Sample] = None
    sample_index: Optional[int] = None

class PerformanceReport(BaseModel):
    accuracy: float
    f1_scores: Dict[str, float]  # per-label F1
    macro_f1: float
    weighted_f1: float
    confusion_matrix: Dict[str, Dict[str, int]]
    total_samples: int
    correct_predictions: int

    class Config:
        json_schema_extra = {
            "example": {
                "accuracy": 0.85,
                "f1_scores": {
                    "payment_intent": 0.87,
                    "payment_request": 0.82,
                    "smart_payment_system_command": 0.90
                },
                "macro_f1": 0.86,
                "weighted_f1": 0.85,
                "confusion_matrix": {
                    "payment_intent": {"payment_intent": 45, "payment_request": 3, "smart_payment_system_command": 2},
                    "payment_request": {"payment_intent": 2, "payment_request": 38, "smart_payment_system_command": 1},
                    "smart_payment_system_command": {"payment_intent": 1, "payment_request": 0, "smart_payment_system_command": 48}
                },
                "total_samples": 140,
                "correct_predictions": 119
            }
        }

class LowConfidenceSample(BaseModel):
    sample: Sample
    prediction: Prediction
    confidence_gap: float  # difference between top 2 predictions

    class Config:
        json_schema_extra = {
            "example": {
                "sample": {"msg": "Can you help me send money?", "label": "payment_intent"},
                "prediction": {
                    "text": "Can you help me send money?",
                    "predicted_label": "payment_request",
                    "confidence": 0.52,
                    "label_probabilities": {
                        "payment_intent": 0.48,
                        "payment_request": 0.52,
                        "smart_payment_system_command": 0.00
                    }
                },
                "confidence_gap": 0.04
            }
        }

class ErrorAnalysis(BaseModel):
    total_errors: int
    error_rate: float
    confusion_patterns: Dict[str, List[str]]  # actual -> list of predicted labels
    most_confused_pairs: List[Tuple[str, str]]  # (actual, predicted) pairs with highest confusion
    error_samples: List[LowConfidenceSample]  # actual error samples for review

    class Config:
        json_schema_extra = {
            "example": {
                "total_errors": 21,
                "error_rate": 0.15,
                "confusion_patterns": {
                    "payment_intent": ["payment_request", "smart_payment_system_command"],
                    "payment_request": ["payment_intent"],
                    "smart_payment_system_command": ["payment_intent"]
                },
                "most_confused_pairs": [
                    ("payment_intent", "payment_request"),
                    ("payment_request", "payment_intent")
                ],
                "error_samples": []
            }
        }

class ConfidenceAnalysis(BaseModel):
    low_confidence_samples: List[LowConfidenceSample]
    confidence_threshold: float
    avg_confidence: float
    confidence_distribution: Dict[str, int]  # confidence ranges -> count

    class Config:
        json_schema_extra = {
            "example": {
                "low_confidence_samples": [],
                "confidence_threshold": 0.8,
                "avg_confidence": 0.87,
                "confidence_distribution": {
                    "0.0-0.2": 2,
                    "0.2-0.4": 5,
                    "0.4-0.6": 8,
                    "0.6-0.8": 15,
                    "0.8-1.0": 110
                }
            }
        }

class ModelAnalysisReport(BaseModel):
    performance: PerformanceReport
    confidence_analysis: ConfidenceAnalysis
    error_analysis: ErrorAnalysis
    model_checkpoint: str
    evaluation_timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "performance": {},
                "confidence_analysis": {},
                "error_analysis": {},
                "model_checkpoint": "3",
                "evaluation_timestamp": "2024-09-23T10:30:00Z"
            }
        }