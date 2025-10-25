from src.payment_classifier.llm.base import BaseLLM
from src.payment_classifier.prompts.base import BasePromptManager
from app.core.schemas.workflow import Sample
from app.core.models.models import LabelConfig
from typing import List, Union
from pydantic import BaseModel
import logging
import json

logger = logging.getLogger(__name__)


class IncorrectSample(BaseModel):
    """Represents an incorrect sample with its index and expected label"""
    index: int
    expected_label: Union[str, int]


class BatchValidationResult(BaseModel):
    """Result of validating a batch of samples - only contains incorrect samples"""
    incorrect_samples: List[IncorrectSample]


class DataValidator:
    """
    Service for validating and correcting labels in memory using LLM.

    This service:
    - Takes a list of samples and validates their labels
    - Returns corrected samples with fixed labels
    - Used by DataGenerator to validate batches before saving
    """

    VALIDATION_PROMPT_KEY = "v2/validation/validate_labels"

    def __init__(
        self,
        llm: BaseLLM,
        prompt_mgr: BasePromptManager,
        label_config: LabelConfig
    ):
        """
        Initialize DataValidator.

        Args:
            llm: Language model for validation
            prompt_mgr: Prompt manager for getting validation prompts
            label_config: Label configuration for understanding labels
        """
        self.llm = llm
        self.prompt_mgr = prompt_mgr
        self.label_config = label_config

    async def validate_and_fix(self, samples: List[Sample], batch_size: int = 30) -> List[Sample]:
        """
        Validate and fix labels for a list of samples in memory.

        Large sample lists are automatically split into smaller batches to ensure
        accurate validation (LLM context limits and attention issues with large inputs).

        Args:
            samples: List of samples to validate
            batch_size: Maximum samples per validation batch (default: 30)

        Returns:
            List of samples with corrected labels (same samples if labels were correct)
        """
        if not samples:
            logger.warning("No samples provided for validation")
            return []

        logger.info(f"Validating {len(samples)} samples (batch_size={batch_size})")

        # Split into batches if needed
        if len(samples) > batch_size:
            logger.info(f"Splitting into {(len(samples) + batch_size - 1) // batch_size} batches for validation")
            return await self._validate_in_batches(samples, batch_size)

        # Small enough to validate in single call
        validation_result = await self._validate_batch(samples, batch_offset=0)

        if not validation_result.incorrect_samples:
            logger.info("All labels are correct")
            return samples

        # Create correction map: index -> corrected label
        correction_map = {
            item.index: item.expected_label
            for item in validation_result.incorrect_samples
        }

        # Apply corrections
        corrected_samples = []
        for i, sample in enumerate(samples):
            if i in correction_map:
                corrected_label = correction_map[i]
                logger.debug(f"Fixed label at index {i}: '{sample.msg[:50]}...' "
                           f"{sample.label} -> {corrected_label}")
                corrected_samples.append(Sample(msg=sample.msg, label=corrected_label))
            else:
                corrected_samples.append(sample)

        logger.info(f"Corrected {len(correction_map)} out of {len(samples)} samples")
        return corrected_samples

    async def _validate_in_batches(self, samples: List[Sample], batch_size: int) -> List[Sample]:
        """
        Validate samples by splitting into smaller batches.

        Args:
            samples: All samples to validate
            batch_size: Size of each validation batch

        Returns:
            List of samples with corrected labels
        """
        correction_map = {}  # Global correction map across all batches

        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch_start = i
            batch_end = min(i + batch_size, len(samples))
            batch = samples[batch_start:batch_end]

            batch_num = (i // batch_size) + 1
            total_batches = (len(samples) + batch_size - 1) // batch_size
            logger.info(f"Validating batch {batch_num}/{total_batches} ({len(batch)} samples, indices {batch_start}-{batch_end-1})")

            # Validate this batch with correct offset
            validation_result = await self._validate_batch(batch, batch_offset=batch_start)

            # Add corrections from this batch to global map
            for item in validation_result.incorrect_samples:
                correction_map[item.index] = item.expected_label

            logger.info(f"Batch {batch_num}: Found {len(validation_result.incorrect_samples)} corrections")

        # Apply all corrections
        corrected_samples = []
        for i, sample in enumerate(samples):
            if i in correction_map:
                corrected_label = correction_map[i]
                logger.debug(f"Fixed label at index {i}: '{sample.msg[:50]}...' "
                           f"{sample.label} -> {corrected_label}")
                corrected_samples.append(Sample(msg=sample.msg, label=corrected_label))
            else:
                corrected_samples.append(sample)

        total_corrections = len(correction_map)
        logger.info(f"Validation complete: Corrected {total_corrections} out of {len(samples)} samples ({total_corrections/len(samples)*100:.1f}%)")
        return corrected_samples

    async def _validate_batch(self, batch: List[Sample], batch_offset: int) -> BatchValidationResult:
        """
        Validate a batch of samples using LLM.

        Args:
            batch: Batch of samples to validate
            batch_offset: Global offset for this batch (for correct indexing)

        Returns:
            BatchValidationResult with list of incorrect samples
        """
        # Get validation prompt
        prompt = self._get_validation_prompt()

        # Build label descriptions for the prompt
        label_descriptions = self._build_label_descriptions(self.label_config)

        # Format batch samples for LLM with their global index
        batch_data = json.dumps([
            {"index": batch_offset + i, "msg": sample.msg, "label": sample.label}
            for i, sample in enumerate(batch)
        ], ensure_ascii=False, indent=2)

        # Create user input with label descriptions and batch data
        user_input = f"""## Label Definitions:
{label_descriptions}

## Samples to Validate:
{batch_data}

Validate each sample and determine if the label is correct. Only output the incorrect samples with their index and expected label."""

        # Call LLM for structured validation
        try:
            result = await self.llm.generate_structured_output(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ],
                BatchValidationResult
            )

            return result
        except Exception as e:
            logger.error(f"Error validating batch: {e}")
            # Return empty list (all correct) if validation fails
            return BatchValidationResult(incorrect_samples=[])

    def _get_validation_prompt(self) -> str:
        """
        Get validation prompt from prompt manager or return default.

        Returns:
            Validation prompt string
        """
        try:
            return self.prompt_mgr.get_prompt(self.VALIDATION_PROMPT_KEY)
        except:
            # Return default prompt if not found
            return """You are a data validation expert. Your task is to validate the correctness of labeled text samples.

For each sample, you must:
1. Read the message carefully
2. Understand the provided label definitions
3. Determine if the assigned label is correct
4. If incorrect, identify the correct label

**IMPORTANT**: Only output samples that have incorrect labels. For each incorrect sample, return:
- index: The position/index of the sample in the given list
- expected_label: The correct label that should be assigned

If all samples are correctly labeled, return an empty list."""

    def _build_label_descriptions(self, label_config: LabelConfig) -> str:
        """
        Build label descriptions for the prompt.

        Args:
            label_config: Label configuration

        Returns:
            Formatted string with label descriptions
        """
        id2label = label_config.get_id2label()

        # Try to get label explanations if available
        try:
            label_explanations = label_config.get_label_explanations()
            descriptions = []
            for label_name in id2label.values():
                explanation = label_explanations.get(label_name, "No description available")
                descriptions.append(f"- {label_name}: {explanation}")
            return "\n".join(descriptions)
        except:
            # Fallback: just list labels
            return "\n".join([f"- {label_name}" for label_name in id2label.values()])
