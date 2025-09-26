import asyncio
import logging
from typing import Dict, List, Optional

from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
from app.core.schemas.analysis import ErrorsByLabel, ErrorCase, ErrorPatternAnalysis

logger = logging.getLogger(__name__)


class ErrorPatternAnalysisService:
    """Service for analyzing error patterns using LLM-based analysis"""

    def __init__(self, llm: BaseLLM, prompt_mgr: BasePromptManager):
        self.llm = llm
        self.prompt_mgr = prompt_mgr

    async def analyze_error_patterns(
        self,
        errors_by_label: Dict[str, ErrorsByLabel],
        label_explanations: Optional[Dict[str, str]] = None
    ) -> List[ErrorPatternAnalysis]:
        """
        Analyze error patterns from grouped errors using LLM analysis.

        Args:
            errors_by_label: Output from ModelAnalyzer._group_errors_by_label()
            label_explanations: Optional explanations for each label

        Returns:
            List of ErrorPatternAnalysis objects, one per error group
        """
        # Extract error groups (expected_label -> predicted_label)
        error_groups = self._extract_error_groups(errors_by_label)

        if not error_groups:
            logger.info("No error patterns found to analyze")
            # Debug: Log what we have in errors_by_label
            total_errors = sum(
                len(label_errors.false_positives) + len(label_errors.false_negatives)
                for label_errors in errors_by_label.values()
            )
            logger.info(f"Total errors in errors_by_label: {total_errors}")
            for label, label_errors in errors_by_label.items():
                logger.info(f"Label {label}: {len(label_errors.false_positives)} FP, {len(label_errors.false_negatives)} FN")
            return []

        logger.info(f"Found {len(error_groups)} error groups to analyze:")
        for pattern_key, error_cases in error_groups.items():
            logger.info(f"  {pattern_key}: {len(error_cases)} cases")

        # Prepare tasks for parallel analysis
        analysis_tasks = []
        for pattern_key, error_cases in error_groups.items():
            if len(error_cases) >= 2:  # Only analyze groups with multiple examples
                logger.info(f"Adding {pattern_key} for analysis ({len(error_cases)} cases)")
                task = self._analyze_single_error_group(
                    pattern_key=pattern_key,
                    error_cases=error_cases,
                    label_explanations=label_explanations or {}
                )
                analysis_tasks.append(task)
            else:
                logger.info(f"Skipping {pattern_key}: only {len(error_cases)} cases (need >= 2)")

        if not analysis_tasks:
            logger.info("No error groups with sufficient examples for analysis (need >= 2 examples per group)")
            return []

        # Run all analyses in parallel
        logger.info(f"Running {len(analysis_tasks)} error pattern analyses in parallel")
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Filter successful results
        successful_analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error pattern analysis {i} failed: {result}")
            elif result:
                successful_analyses.append(result)
                logger.info(f"Successfully completed analysis {i}: {result.predicted_label} -> {result.expected_label}")
            else:
                logger.warning(f"Error pattern analysis {i} returned None/empty result")

        logger.info(f"Successfully completed {len(successful_analyses)} out of {len(results)} error pattern analyses")
        return successful_analyses

    def _extract_error_groups(self, errors_by_label: Dict[str, ErrorsByLabel]) -> Dict[str, List[ErrorCase]]:
        """
        Extract error groups from errors_by_label format.
        Groups errors by (expected_label -> predicted_label) pattern.
        """
        error_groups = {}

        # Process false negatives (cases that should be this label but weren't)
        for expected_label, label_errors in errors_by_label.items():
            for error_case in label_errors.false_negatives:
                pattern_key = f"{expected_label}->{error_case.predicted_label}"
                if pattern_key not in error_groups:
                    error_groups[pattern_key] = []
                error_groups[pattern_key].append(error_case)

        return error_groups

    async def _analyze_single_error_group(
        self,
        pattern_key: str,
        error_cases: List[ErrorCase],
        label_explanations: Dict[str, str]
    ) -> Optional[ErrorPatternAnalysis]:
        """
        Analyze a single error group using LLM.

        Args:
            pattern_key: Format "expected_label->predicted_label"
            error_cases: List of ErrorCase objects for this pattern
            label_explanations: Explanations for labels

        Returns:
            ErrorPatternAnalysis or None if analysis fails
        """
        try:
            # Parse pattern key
            expected_label, predicted_label = pattern_key.split('->', 1)

            # Format test cases for the prompt
            test_cases = self._format_test_cases(error_cases)

            # Get label explanation
            label_explanation = self._get_label_explanation(expected_label, predicted_label, label_explanations)

            # Get the prompt template
            prompt_template = self.prompt_mgr.get_prompt("analyze/error_pattern_detect")

            # Format the prompt
            formatted_prompt = prompt_template.format(
                predicted_label=predicted_label,
                expected_label=expected_label,
                label_explanation=label_explanation,
                test_cases=test_cases
            )

            # Call LLM for structured analysis
            logger.debug(f"Analyzing error pattern: {pattern_key} with {len(error_cases)} examples")

            messages = [
                {"role": "system", "content": formatted_prompt}
            ]

            # Use generate_structured_output to get properly formatted response
            result = await self.llm.generate_structured_output(
                messages,
                ErrorPatternAnalysis
            )

            # The result should be an ErrorPatternAnalysis instance
            # Just set the labels that we know from the context
            result.predicted_label = predicted_label
            result.expected_label = expected_label

            return result

        except Exception as e:
            logger.error(f"Error analyzing pattern {pattern_key}: {e}")
            return None

    def _format_test_cases(self, error_cases: List[ErrorCase]) -> str:
        """Format error cases for the prompt"""
        formatted_cases = []

        for i, case in enumerate(error_cases, 1):
            formatted_cases.append(f"{i}. \"{case.input.msg}\"")

        return "\n".join(formatted_cases)

    def _get_label_explanation(
        self,
        expected_label: str,
        predicted_label: str,
        label_explanations: Dict[str, str]
    ) -> str:
        """Get explanation for the labels involved in the error"""
        explanations = []

        if expected_label in label_explanations:
            explanations.append(f"Expected label '{expected_label}': {label_explanations[expected_label]}")

        if predicted_label in label_explanations:
            explanations.append(f"Predicted label '{predicted_label}': {label_explanations[predicted_label]}")

        if not explanations:
            return f"Analyzing misclassification from '{expected_label}' to '{predicted_label}'"

        return "\n".join(explanations)

