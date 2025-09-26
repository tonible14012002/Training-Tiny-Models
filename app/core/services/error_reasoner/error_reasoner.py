import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from app.core.schemas.analysis import TestCase, ErrorBucket, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ErrorReason:
    """Represents a specific reason why a misclassification occurred"""
    reason_type: str  # e.g., "keyword_confusion", "context_missing", "structural_complexity"
    description: str  # Human-readable explanation of the issue
    confidence: float  # How confident we are in this reason (0-1)
    evidence: Dict  # Supporting evidence (e.g., conflicting keywords, missing context clues)
    data_generation_prompt: str  # Prompt fragment to guide LLM data generation


@dataclass
class MisclassificationAnalysis:
    """Analysis results for a single misclassified test case"""
    test_case: TestCase
    primary_reasons: List[ErrorReason]  # Most likely reasons
    secondary_reasons: List[ErrorReason]  # Possible but less likely reasons
    overall_confidence: float  # Overall confidence in the analysis


class ErrorReasoner:
    """
    Service responsible for analyzing misclassification patterns and generating
    detailed reasoning about why specific errors occurred. This reasoning is then
    used to create targeted prompts for synthetic data generation.

    The ErrorReasoner bridges the gap between raw error detection (ModelAnalyzer)
    and strategic data generation (DataGenerator) by providing detailed insights
    into the root causes of misclassifications.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_reason_templates()

    def _initialize_reason_templates(self):
        """Initialize templates for different types of error reasons"""
        self.reason_templates = {
            "keyword_confusion": {
                "description_template": "Model confused by keyword '{keyword}' in wrong context",
                "prompt_template": "Generate examples where '{keyword}' appears in {correct_context} context to reduce confusion"
            },
            "missing_context": {
                "description_template": "Model lacks contextual understanding for intent '{intent}'",
                "prompt_template": "Create examples with rich context clues for '{intent}' classification"
            },
            "structural_complexity": {
                "description_template": "Complex sentence structure hindered intent detection",
                "prompt_template": "Generate examples with varied sentence structures for intent '{intent}'"
            },
            "semantic_similarity": {
                "description_template": "Semantic similarity between '{true_intent}' and '{predicted_intent}' caused confusion",
                "prompt_template": "Create examples that clearly distinguish '{true_intent}' from '{predicted_intent}'"
            },
            "ambiguous_phrasing": {
                "description_template": "Ambiguous phrasing led to intent misinterpretation",
                "prompt_template": "Generate unambiguous examples for '{intent}' with clear intent signals"
            }
        }

    def analyze_misclassifications(
        self,
        evaluation_result: EvaluationResult,
        error_buckets: Optional[List[ErrorBucket]] = None
    ) -> List[MisclassificationAnalysis]:
        """
        Analyze all misclassified test cases and provide detailed reasoning.

        Args:
            evaluation_result: Results from ModelAnalyzer
            error_buckets: Optional pre-identified error patterns

        Returns:
            List of detailed misclassification analyses
        """
        if not evaluation_result.test_cases:
            self.logger.warning("No test cases available for misclassification analysis")
            return []

        # Filter misclassified cases
        misclassified_cases = [
            tc for tc in evaluation_result.test_cases
            if tc.prediction and tc.prediction.label != tc.true_label
        ]

        self.logger.info(f"Analyzing {len(misclassified_cases)} misclassified cases")

        analyses = []
        for test_case in misclassified_cases:
            analysis = self._analyze_single_case(test_case, error_buckets)
            analyses.append(analysis)

        return analyses

    def _analyze_single_case(
        self,
        test_case: TestCase,
        error_buckets: Optional[List[ErrorBucket]] = None
    ) -> MisclassificationAnalysis:
        """
        Analyze a single misclassified test case to determine likely reasons.

        This is a placeholder implementation that will be expanded with:
        - Text analysis for keyword conflicts
        - Semantic similarity analysis
        - Structural complexity assessment
        - Context adequacy evaluation
        """
        # Placeholder implementation - will be expanded
        primary_reasons = []
        secondary_reasons = []

        # Basic analysis based on available information
        text = test_case.input.msg.lower()
        true_label = test_case.true_label
        predicted_label = test_case.prediction.label if test_case.prediction else "unknown"
        confidence = test_case.prediction.prob if test_case.prediction else 0.0

        # Simple heuristic-based reasoning (to be improved)
        if confidence > 0.8:
            # High confidence wrong answer suggests keyword confusion
            reason = ErrorReason(
                reason_type="keyword_confusion",
                description=f"High confidence misclassification suggests keyword confusion between {true_label} and {predicted_label}",
                confidence=0.7,
                evidence={"high_confidence": confidence, "text_length": len(text.split())},
                data_generation_prompt=f"Generate examples that distinguish {true_label} from {predicted_label} using different vocabulary"
            )
            primary_reasons.append(reason)

        if len(text.split()) > 15:
            # Long text suggests structural complexity
            reason = ErrorReason(
                reason_type="structural_complexity",
                description="Complex sentence structure may have hindered classification",
                confidence=0.6,
                evidence={"text_length": len(text.split()), "complexity_indicators": ["long_sentence"]},
                data_generation_prompt=f"Create examples with varied sentence structures for {true_label} intent"
            )
            secondary_reasons.append(reason)

        # Calculate overall confidence based on individual reason confidences
        if primary_reasons:
            overall_confidence = max(reason.confidence for reason in primary_reasons)
        else:
            overall_confidence = 0.5  # Neutral confidence when no strong reasons found

        return MisclassificationAnalysis(
            test_case=test_case,
            primary_reasons=primary_reasons,
            secondary_reasons=secondary_reasons,
            overall_confidence=overall_confidence
        )

    def generate_data_generation_prompts(
        self,
        analyses: List[MisclassificationAnalysis],
        max_prompts: int = 10
    ) -> List[str]:
        """
        Generate targeted prompts for synthetic data generation based on error analysis.

        Args:
            analyses: Results from analyze_misclassifications
            max_prompts: Maximum number of prompts to generate

        Returns:
            List of targeted prompts for data generation
        """
        prompts = []

        # Group reasons by type and frequency
        reason_counts = {}
        all_reasons = []

        for analysis in analyses:
            all_reasons.extend(analysis.primary_reasons)
            all_reasons.extend(analysis.secondary_reasons)

        for reason in all_reasons:
            reason_type = reason.reason_type
            if reason_type not in reason_counts:
                reason_counts[reason_type] = []
            reason_counts[reason_type].append(reason)

        # Generate prompts for most common reason types
        for reason_type, reasons in sorted(reason_counts.items(), key=lambda x: len(x[1]), reverse=True):
            if len(prompts) >= max_prompts:
                break

            # Use the highest confidence reason of this type
            best_reason = max(reasons, key=lambda r: r.confidence)
            prompts.append(best_reason.data_generation_prompt)

        return prompts[:max_prompts]

    def get_error_summary(self, analyses: List[MisclassificationAnalysis]) -> Dict:
        """
        Generate a summary of error patterns across all analyses.

        Returns:
            Dictionary with error pattern statistics and insights
        """
        if not analyses:
            return {"total_cases": 0, "reason_distribution": {}, "insights": []}

        reason_distribution = {}
        confidence_scores = []

        for analysis in analyses:
            confidence_scores.append(analysis.overall_confidence)

            for reason in analysis.primary_reasons + analysis.secondary_reasons:
                reason_type = reason.reason_type
                reason_distribution[reason_type] = reason_distribution.get(reason_type, 0) + 1

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        # Generate insights
        insights = []
        if reason_distribution:
            most_common = max(reason_distribution.items(), key=lambda x: x[1])
            insights.append(f"Most common error type: {most_common[0]} ({most_common[1]} cases)")

        if avg_confidence < 0.5:
            insights.append("Low confidence in error reasoning - may need more sophisticated analysis")
        elif avg_confidence > 0.8:
            insights.append("High confidence in error patterns - targeted data generation recommended")

        return {
            "total_cases": len(analyses),
            "average_confidence": avg_confidence,
            "reason_distribution": reason_distribution,
            "insights": insights
        }

    # Placeholder methods for future implementation

    def _analyze_keyword_conflicts(self, test_case: TestCase) -> List[ErrorReason]:
        """Analyze potential keyword-based confusion (to be implemented)"""
        # TODO: Implement keyword conflict analysis
        # - Extract key terms from text
        # - Check for terms associated with different intents
        # - Analyze semantic similarity between conflicting terms
        pass

    def _analyze_context_adequacy(self, test_case: TestCase) -> List[ErrorReason]:
        """Analyze whether sufficient context was provided (to be implemented)"""
        # TODO: Implement context analysis
        # - Check for context clues in the text
        # - Identify missing information that could clarify intent
        # - Assess sentence completeness and clarity
        pass

    def _analyze_semantic_similarity(self, test_case: TestCase) -> List[ErrorReason]:
        """Analyze semantic similarity between true and predicted intents (to be implemented)"""
        # TODO: Implement semantic similarity analysis
        # - Compare embedding similarity between intent classes
        # - Identify potential confusion boundaries
        # - Suggest disambiguation strategies
        pass

    def _analyze_structural_patterns(self, test_case: TestCase) -> List[ErrorReason]:
        """Analyze sentence structure and grammatical patterns (to be implemented)"""
        # TODO: Implement structural analysis
        # - Parse sentence structure
        # - Identify complex grammatical constructions
        # - Assess impact on intent recognition
        pass