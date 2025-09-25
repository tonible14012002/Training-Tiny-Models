import logging
from typing import List, Dict, Set, Optional
from datasets import Dataset

from app.core.services.trainer.trainer import TrainerService
from app.core.services.data_manager.data_manager import DataManager
from app.core.schemas.analysis import (
    EvaluationResult,
    OverallMetrics,
    LabelMetrics,
    TestCase,
    Prediction,
    ErrorBucket,
    OpenIntentAnalysis,
    MisclassifiedOpenIntent
)
from app.core.schemas.workflow import Sample
from src.payment_classifier.inference.adb_inference import ADBModelInference

logger = logging.getLogger(__name__)

class ModelAnalyzer:
    '''
    Analyze and provide insights on machine learning models using ADB inference.
    '''

    def __init__(self, trainer_service: TrainerService, data_manager: DataManager):
        self.trainer_service = trainer_service
        self.data_manager = data_manager
        self.adb_inferencer: Optional[ADBModelInference] = None

    def load_model(self, checkpoint_path: str) -> None:
        """Load the ADB model from checkpoint path"""
        try:
            self.adb_inferencer = ADBModelInference(peft_path=checkpoint_path)
            logger.info(f"Loaded ADB model from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {e}")
            raise

    def analyze_model(
        self,
        evaluation_dataset: Dataset,
        open_intent_samples: Optional[List[str]] = None,
        include_test_cases: bool = False
    ) -> EvaluationResult:
        """
        Comprehensive analysis of the trained model using ADB inference.

        Args:
            evaluation_dataset: Dataset with 'msg' and 'label' columns
            open_intent_samples: List of open intent samples to test OOD detection
            include_test_cases: Whether to include individual test case results

        Returns:
            EvaluationResult with comprehensive metrics
        """
        if self.adb_inferencer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Analyzing model on {len(evaluation_dataset)} evaluation samples")

        # Analyze known intents using ADB evaluation
        adb_results = self.adb_inferencer.evaluate_with_adb(evaluation_dataset)

        # Convert to our schema format
        overall_metrics = OverallMetrics(**adb_results["overall"])

        per_label_metrics = {}
        for label, metrics in adb_results["per_label"].items():
            per_label_metrics[label] = LabelMetrics(**metrics)

        # Create test cases if requested
        test_cases = None
        if include_test_cases:
            test_cases = self._create_test_cases(evaluation_dataset, adb_results)

        # Analyze open intent samples if provided
        open_intent_analysis = None
        if open_intent_samples:
            open_intent_results = self._analyze_open_intents(open_intent_samples)
            logger.info(f"Open intent analysis: {len(open_intent_samples)} samples, "
                       f"{open_intent_results['detected_as_unknown']} detected as unknown "
                       f"({open_intent_results['unknown_rate']:.2%})")

            # Convert to schema format
            misclassified = [
                MisclassifiedOpenIntent(**item) for item in open_intent_results["misclassified"]
            ]

            open_intent_analysis = OpenIntentAnalysis(
                total_samples=open_intent_results["total_samples"],
                detected_as_unknown=open_intent_results["detected_as_unknown"],
                unknown_rate=open_intent_results["unknown_rate"],
                false_positive_rate=open_intent_results["false_positive_rate"],
                misclassified=misclassified
            )

        result = EvaluationResult(
            overall=overall_metrics,
            per_label=per_label_metrics,
            adb_info=adb_results.get("adb_info"),
            test_cases=test_cases,
            open_intent_analysis=open_intent_analysis
        )

        return result

    def _create_test_cases(self, evaluation_dataset: Dataset, adb_results: dict) -> List[TestCase]:
        """Create test cases from evaluation results"""
        test_cases = []

        # Get predictions for all samples
        batch_size = 32
        all_predictions = []

        for i in range(0, len(evaluation_dataset), batch_size):
            batch = evaluation_dataset[i:i+batch_size]
            texts = batch['msg']
            predictions = self.adb_inferencer.predict_with_adb(texts)
            all_predictions.extend(predictions)

        # Create test case objects
        for i, (sample_data, pred_data) in enumerate(zip(evaluation_dataset, all_predictions)):
            sample = Sample(
                msg=sample_data['msg'],
                label=self.adb_inferencer.id2label[sample_data['label']]
            )

            prediction = Prediction(
                label=pred_data["label"],
                prob=pred_data["prob"],
                dis=pred_data.get("dis"),
                closest=pred_data.get("closest")
            )

            test_case = TestCase(
                input=sample,
                true_label=self.adb_inferencer.id2label[sample_data['label']],
                prediction=prediction
            )
            test_cases.append(test_case)

        return test_cases

    def _analyze_open_intents(self, open_intent_samples: List[str]) -> Dict:
        """Analyze open intent detection capability"""
        if self.adb_inferencer is None:
            raise ValueError("Model not loaded")

        predictions = self.adb_inferencer.predict_with_adb(open_intent_samples)

        detected_as_unknown = sum(1 for pred in predictions if pred["label"] == "Unknown")
        total_samples = len(open_intent_samples)
        unknown_rate = detected_as_unknown / total_samples if total_samples > 0 else 0.0

        # Analyze misclassified open intents
        misclassified = []
        for i, pred in enumerate(predictions):
            if pred["label"] != "Unknown":
                misclassified.append({
                    "text": open_intent_samples[i],
                    "predicted_as": pred["label"],
                    "confidence": pred["prob"],
                    "distance": pred.get("dis", 0.0)
                })

        return {
            "total_samples": total_samples,
            "detected_as_unknown": detected_as_unknown,
            "unknown_rate": unknown_rate,
            "misclassified": misclassified,
            "false_positive_rate": (total_samples - detected_as_unknown) / total_samples if total_samples > 0 else 0.0
        }

    def analyze_errors(self, evaluation_result: EvaluationResult) -> List[ErrorBucket]:
        """
        Analyze errors and categorize them into error buckets.

        Args:
            evaluation_result: Result from analyze_model()

        Returns:
            List of relevant error buckets based on the analysis
        """
        if not evaluation_result.test_cases:
            logger.warning("No test cases available for error analysis. Run analyze_model with include_test_cases=True")
            return []

        # Analyze error patterns
        error_patterns = self._categorize_errors(evaluation_result.test_cases)

        # Map to error buckets
        relevant_buckets = self._map_to_error_buckets(error_patterns, evaluation_result.overall)

        return relevant_buckets

    def _categorize_errors(self, test_cases: List[TestCase]) -> Dict:
        """Categorize errors based on test case analysis"""
        patterns = {
            "keyword_reliance": 0,
            "simple_intent_miss": 0,
            "complex_structure": 0,
            "ambiguous_cases": 0,
            "focus_issues": 0
        }

        total_errors = 0

        for test_case in test_cases:
            if test_case.prediction and test_case.prediction.label != test_case.true_label:
                total_errors += 1

                # Simple heuristics for error categorization
                msg = test_case.input.msg.lower()
                msg_len = len(msg.split())

                # Simple intent miss - short, clear messages
                if msg_len <= 5 and any(word in msg for word in ['pay', 'send', 'request', 'money']):
                    patterns["simple_intent_miss"] += 1

                # Complex structure - long sentences with multiple clauses
                elif msg_len > 10 or ',' in msg or 'if' in msg or 'when' in msg:
                    patterns["complex_structure"] += 1

                # Ambiguous cases - could be multiple intents
                elif any(word in msg for word in ['maybe', 'could', 'might', 'possibly']):
                    patterns["ambiguous_cases"] += 1

                # Default to keyword reliance or focus issues
                else:
                    if test_case.prediction.prob > 0.8:  # High confidence wrong answer
                        patterns["keyword_reliance"] += 1
                    else:
                        patterns["focus_issues"] += 1

        patterns["total_errors"] = total_errors
        return patterns

    def _map_to_error_buckets(self, patterns: Dict, overall_metrics: OverallMetrics) -> List[ErrorBucket]:
        """Map error patterns to relevant error buckets"""
        # Import the example buckets
        from app.core.schemas.analysis import EXAMPLE_ERROR_BUCKETS

        relevant_buckets = []

        # Add buckets based on error patterns
        if patterns["simple_intent_miss"] > patterns["total_errors"] * 0.2:  # >20% of errors
            relevant_buckets.append(next(bucket for bucket in EXAMPLE_ERROR_BUCKETS
                                       if bucket.name == "Miss detecting simple intent"))

        if patterns["keyword_reliance"] > patterns["total_errors"] * 0.3:  # >30% of errors
            relevant_buckets.append(next(bucket for bucket in EXAMPLE_ERROR_BUCKETS
                                       if bucket.name == "Over reliance on keywords"))

        if patterns["complex_structure"] > patterns["total_errors"] * 0.2:  # >20% of errors
            relevant_buckets.append(next(bucket for bucket in EXAMPLE_ERROR_BUCKETS
                                       if bucket.name == "Complex sentence structure"))

        if patterns["focus_issues"] > patterns["total_errors"] * 0.15:  # >15% of errors
            relevant_buckets.append(next(bucket for bucket in EXAMPLE_ERROR_BUCKETS
                                       if bucket.name == "Miss focus on important word"))

        if overall_metrics.accuracy < 0.7:  # Low overall accuracy
            relevant_buckets.append(next(bucket for bucket in EXAMPLE_ERROR_BUCKETS
                                       if bucket.name == "Ambiguous intent"))

        return relevant_buckets

    def generate_analysis_report(self, evaluation_result: EvaluationResult) -> str:
        """Generate a comprehensive analysis report"""
        report = []

        # Overall metrics
        overall = evaluation_result.overall
        report.append("=== MODEL ANALYSIS REPORT ===\n")
        report.append(f"Overall Accuracy: {overall.accuracy:.2%}")
        report.append(f"Coverage: {overall.coverage:.2%}")
        report.append(f"Unknown Rate: {overall.unknown_rate:.2%}")
        report.append(f"Macro F1-Score: {overall.macro_f1:.3f}")
        report.append(f"Total Samples: {overall.total_samples}")
        report.append("")

        # Per-label metrics
        report.append("=== PER-LABEL METRICS ===")
        for label, metrics in evaluation_result.per_label.items():
            report.append(f"\n{label.upper()}:")
            report.append(f"  Accuracy: {metrics.accuracy:.2%}")
            report.append(f"  Precision: {metrics.precision:.3f}")
            report.append(f"  Recall: {metrics.recall:.3f}")
            report.append(f"  F1-Score: {metrics.f1_score:.3f}")
            report.append(f"  Coverage: {metrics.coverage:.2%}")
            report.append(f"  Samples: {metrics.samples}")

        # Open intent analysis
        if evaluation_result.open_intent_analysis:
            oia = evaluation_result.open_intent_analysis
            report.append("\n=== OPEN INTENT ANALYSIS ===")
            report.append(f"Total Open Intent Samples: {oia.total_samples}")
            report.append(f"Detected as Unknown: {oia.detected_as_unknown}")
            report.append(f"Unknown Detection Rate: {oia.unknown_rate:.2%}")
            report.append(f"False Positive Rate: {oia.false_positive_rate:.2%}")

            if oia.misclassified:
                report.append(f"\nMisclassified Open Intents ({len(oia.misclassified)}):")
                for i, misc in enumerate(oia.misclassified[:5]):  # Show first 5
                    report.append(f"  {i+1}. '{misc.text}' -> {misc.predicted_as} (conf: {misc.confidence:.3f})")
                if len(oia.misclassified) > 5:
                    report.append(f"  ... and {len(oia.misclassified) - 5} more")

        # ADB info
        if evaluation_result.adb_info and "radii" in evaluation_result.adb_info:
            report.append("\n=== ADB RADII ===")
            for label_id, radius in evaluation_result.adb_info["radii"].items():
                label_name = evaluation_result.adb_info["labels"][int(label_id)]
                report.append(f"{label_name}: {radius:.4f}")

        return "\n".join(report)