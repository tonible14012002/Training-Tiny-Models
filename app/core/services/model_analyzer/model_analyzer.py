import logging
from typing import List, Dict, Optional
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
    MisclassifiedOpenIntent,
    ErrorCase,
    ErrorsByLabel
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
        include_test_cases: bool = False,
        use_comprehensive_unknown_evaluation: bool = True
    ) -> EvaluationResult:
        """
        Comprehensive analysis of the trained model using ADB inference.

        Args:
            evaluation_dataset: Dataset with 'msg' and 'label' columns
            open_intent_samples: List of open intent samples to test OOD detection
            include_test_cases: Whether to include individual test case results
            use_comprehensive_unknown_evaluation: If True and open_intent_samples provided,
                                                   use comprehensive evaluation that treats Unknown
                                                   as a proper label tested on both datasets

        Returns:
            EvaluationResult with comprehensive metrics
        """
        if self.adb_inferencer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Analyzing model on {len(evaluation_dataset)} evaluation samples")

        # Choose evaluation method based on parameters
        if open_intent_samples and use_comprehensive_unknown_evaluation:
            # Use comprehensive evaluation that treats Unknown as a proper label
            logger.info(f"Using comprehensive Unknown evaluation with {len(open_intent_samples)} unknown samples")
            adb_results = self.adb_inferencer.evaluate_with_unknown_intents(evaluation_dataset, open_intent_samples)
        else:
            # Use traditional evaluation on known intents only
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

        # Group errors by label for convenience (including unknown misclassified samples)
        errors_by_label = self._group_errors_by_label(
            evaluation_dataset=evaluation_dataset,
            open_intent_samples=open_intent_samples
        )

        # Analyze open intent samples if provided (only if not using comprehensive evaluation)
        open_intent_analysis = None
        if open_intent_samples and not use_comprehensive_unknown_evaluation:
            open_intent_results = self._analyze_open_intents(open_intent_samples)
            logger.info(f"Open intent analysis: {len(open_intent_samples)} samples, "
                       f"{open_intent_results['detected_as_unknown']} detected as unknown "
                       f"({open_intent_results['unknown_rate']:.2%})")

            # Convert to schema format
            try:
                misclassified = [
                    MisclassifiedOpenIntent(**item) for item in open_intent_results["misclassified"]
                ]
                logger.info(f"Successfully converted {len(misclassified)} misclassified items to schema format")
            except Exception as e:
                logger.error(f"Error converting misclassified items to schema format: {e}")
                logger.error(f"Raw misclassified data: {open_intent_results['misclassified']}")
                misclassified = []  # Fallback to empty list

            open_intent_analysis = OpenIntentAnalysis(
                total_samples=open_intent_results["total_samples"],
                detected_as_unknown=open_intent_results["detected_as_unknown"],
                unknown_rate=open_intent_results["unknown_rate"],
                false_positive_rate=open_intent_results["false_positive_rate"],
                misclassified=misclassified
            )
        elif open_intent_samples and use_comprehensive_unknown_evaluation:
            # Extract open intent analysis from comprehensive results
            if "unknown_analysis" in adb_results:
                unknown_analysis = adb_results["unknown_analysis"]
                # Create minimal open intent analysis from comprehensive results
                open_intent_analysis = OpenIntentAnalysis(
                    total_samples=len(open_intent_samples),
                    detected_as_unknown=unknown_analysis["true_positives"],
                    unknown_rate=unknown_analysis["unknown_detection_rate"],
                    false_positive_rate=unknown_analysis["false_positive_rate"],
                    misclassified=[]  # Could be enhanced to extract from comprehensive results if needed
                )

        # Pass through unknown_analysis if using comprehensive evaluation
        unknown_analysis = adb_results.get("unknown_analysis") if use_comprehensive_unknown_evaluation and open_intent_samples else None

        result = EvaluationResult(
            overall=overall_metrics,
            per_label=per_label_metrics,
            adb_info=adb_results.get("adb_info"),
            test_cases=test_cases,
            open_intent_analysis=open_intent_analysis,
            unknown_analysis=unknown_analysis,
            errors_by_label=errors_by_label
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

    def _group_errors_by_label(
        self,
        evaluation_dataset: Dataset,
        open_intent_samples: Optional[List[str]] = None
    ) -> Dict[str, ErrorsByLabel]:
        """Group false positives and false negatives by label, including unknown misclassified samples"""
        if self.adb_inferencer is None:
            raise ValueError("Model not loaded")

        # Get all labels in the dataset
        all_labels = set()
        for sample in evaluation_dataset:
            true_label = self.adb_inferencer.id2label[sample['label']]
            all_labels.add(true_label)

        # Initialize error groups for each label
        errors_by_label = {}
        for label in all_labels:
            errors_by_label[label] = ErrorsByLabel(false_positives=[], false_negatives=[])

        # Always include "Unknown" if it exists in the model
        if "Unknown" in self.adb_inferencer.label2id:
            errors_by_label["Unknown"] = ErrorsByLabel(false_positives=[], false_negatives=[])

        # Process evaluation dataset (known intents)
        batch_size = 32
        all_predictions = []

        for i in range(0, len(evaluation_dataset), batch_size):
            batch = evaluation_dataset[i:i+batch_size]
            texts = batch['msg']
            predictions = self.adb_inferencer.predict_with_adb(texts)
            all_predictions.extend(predictions)

        # Group errors from evaluation dataset
        for i, (sample_data, pred_data) in enumerate(zip(evaluation_dataset, all_predictions)):
            true_label = self.adb_inferencer.id2label[sample_data['label']]
            predicted_label = pred_data["label"]

            # Skip correct predictions
            if true_label == predicted_label:
                continue

            # Create error case
            error_case = ErrorCase(
                input=Sample(
                    msg=sample_data['msg'],
                    label=true_label
                ),
                true_label=true_label,
                predicted_label=predicted_label,
                confidence=pred_data["prob"],
                distance=pred_data.get("dis")
            )

            # Add as false negative for true label
            if true_label in errors_by_label:
                errors_by_label[true_label].false_negatives.append(error_case)

            # Add as false positive for predicted label
            if predicted_label in errors_by_label:
                errors_by_label[predicted_label].false_positives.append(error_case)

        # Process open intent samples if provided
        if open_intent_samples and "Unknown" in errors_by_label:
            open_predictions = self.adb_inferencer.predict_with_adb(open_intent_samples)

            for i, pred_data in enumerate(open_predictions):
                predicted_label = pred_data["label"]

                # Skip correct predictions (should be "Unknown")
                if predicted_label == "Unknown":
                    continue

                # Create error case for misclassified unknown sample
                error_case = ErrorCase(
                    input=Sample(
                        msg=open_intent_samples[i],
                        label="Unknown"  # True label for open intent samples
                    ),
                    true_label="Unknown",
                    predicted_label=predicted_label,
                    confidence=pred_data["prob"],
                    distance=pred_data.get("dis")
                )

                # Add as false negative for Unknown (should have been Unknown but wasn't)
                errors_by_label["Unknown"].false_negatives.append(error_case)

                # Add as false positive for predicted label (was predicted as this label but shouldn't be)
                if predicted_label in errors_by_label:
                    errors_by_label[predicted_label].false_positives.append(error_case)

        return errors_by_label

    def group_errors_by_pattern(
        self,
        evaluation_dataset: Dataset,
        open_intent_samples: Optional[List[str]] = None
    ) -> Dict[str, List[ErrorCase]]:
        """
        Group all error cases by (expected_label, predicted_label) tuples.

        Args:
            evaluation_dataset: Dataset with 'msg' and 'label' columns
            open_intent_samples: List of open intent samples (should be classified as Unknown)

        Returns:
            Dict where keys are "expected_label->predicted_label" and values are lists of ErrorCase
        """
        if self.adb_inferencer is None:
            raise ValueError("Model not loaded")

        # Dictionary to group errors by (expected, predicted) tuple
        error_patterns = {}

        # Process evaluation dataset (known intents)
        batch_size = 32
        all_predictions = []

        for i in range(0, len(evaluation_dataset), batch_size):
            batch = evaluation_dataset[i:i+batch_size]
            texts = batch['msg']
            predictions = self.adb_inferencer.predict_with_adb(texts)
            all_predictions.extend(predictions)

        # Group errors from evaluation dataset
        for i, (sample_data, pred_data) in enumerate(zip(evaluation_dataset, all_predictions)):
            true_label = self.adb_inferencer.id2label[sample_data['label']]
            predicted_label = pred_data["label"]

            # Skip correct predictions
            if true_label == predicted_label:
                continue

            # Create error case
            error_case = ErrorCase(
                input=Sample(
                    msg=sample_data['msg'],
                    label=true_label
                ),
                true_label=true_label,
                predicted_label=predicted_label,
                confidence=pred_data["prob"],
                distance=pred_data.get("dis")
            )

            # Group by (expected, predicted) tuple
            pattern_key = f"{true_label}->{predicted_label}"
            if pattern_key not in error_patterns:
                error_patterns[pattern_key] = []

            error_patterns[pattern_key].append(error_case)

        # Process open intent samples if provided
        if open_intent_samples:
            open_predictions = self.adb_inferencer.predict_with_adb(open_intent_samples)

            for i, pred_data in enumerate(open_predictions):
                predicted_label = pred_data["label"]

                # Skip correct predictions (should be "Unknown")
                if predicted_label == "Unknown":
                    continue

                # Create error case for misclassified unknown sample
                error_case = ErrorCase(
                    input=Sample(
                        msg=open_intent_samples[i],
                        label="Unknown"  # True label for open intent samples
                    ),
                    true_label="Unknown",
                    predicted_label=predicted_label,
                    confidence=pred_data["prob"],
                    distance=pred_data.get("dis")
                )

                # Group by (expected, predicted) tuple
                pattern_key = f"Unknown->{predicted_label}"
                if pattern_key not in error_patterns:
                    error_patterns[pattern_key] = []

                error_patterns[pattern_key].append(error_case)

        return error_patterns

    def get_error_patterns_from_result(self, errors_by_label: Dict[str, ErrorsByLabel]) -> Dict[str, List[ErrorCase]]:
        """
        Extract error patterns from existing errors_by_label result.

        Args:
            errors_by_label: Result from _group_errors_by_label()

        Returns:
            Dict where keys are "expected_label->predicted_label" and values are lists of ErrorCase
        """
        error_patterns = {}

        # Extract error cases from errors_by_label
        for _, label_errors in errors_by_label.items():
            # Process false negatives (cases that should be this label but weren't)
            for error_case in label_errors.false_negatives:
                pattern_key = f"{error_case.true_label}->{error_case.predicted_label}"
                if pattern_key not in error_patterns:
                    error_patterns[pattern_key] = []
                error_patterns[pattern_key].append(error_case)

        return error_patterns

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

        result = {
            "total_samples": total_samples,
            "detected_as_unknown": detected_as_unknown,
            "unknown_rate": unknown_rate,
            "misclassified": misclassified,
            "false_positive_rate": (total_samples - detected_as_unknown) / total_samples if total_samples > 0 else 0.0
        }

        # Add logging to track the data
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Open intent analysis results: {total_samples} total, {detected_as_unknown} detected as unknown, {len(misclassified)} misclassified")
        if misclassified:
            logger.info(f"Sample misclassified items: {misclassified[:2]}")

        return result

    def generate_analysis_report(self, evaluation_result: EvaluationResult) -> str:
        """Generate a comprehensive analysis report"""
        report = []

        # Overall metrics
        overall = evaluation_result.overall
        report.append("=== MODEL ANALYSIS REPORT ===\n")
        report.append(f"Overall Accuracy: {overall.accuracy:.2%}")
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

        # Unknown analysis (if using comprehensive evaluation)
        if hasattr(evaluation_result, 'unknown_analysis') and evaluation_result.unknown_analysis:
            ua = evaluation_result.unknown_analysis
            report.append("\n=== COMPREHENSIVE UNKNOWN ANALYSIS ===")
            report.append(f"Unknown Detection Performance:")
            report.append(f"  True Positives: {ua['true_positives']} (Unknown correctly identified)")
            report.append(f"  False Positives: {ua['false_positives']} (Known incorrectly as Unknown)")
            report.append(f"  True Negatives: {ua['true_negatives']} (Known correctly NOT as Unknown)")
            report.append(f"  False Negatives: {ua['false_negatives']} (Unknown incorrectly as known)")
            report.append(f"  Unknown Detection Rate: {ua['unknown_detection_rate']:.2%}")
            report.append(f"  False Positive Rate: {ua['false_positive_rate']:.2%}")
            report.append(f"  Precision: {ua['precision']:.3f}")
            report.append(f"  Recall: {ua['recall']:.3f}")
            report.append(f"  F1-Score: {ua['f1_score']:.3f}")

        # ADB info
        if evaluation_result.adb_info and "radii" in evaluation_result.adb_info:
            report.append("\n=== ADB RADII ===")
            for label_id, radius in evaluation_result.adb_info["radii"].items():
                label_name = evaluation_result.adb_info["labels"][int(label_id)]
                report.append(f"{label_name}: {radius:.4f}")

        return "\n".join(report)