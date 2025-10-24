from typing import List, Tuple
from pydantic import BaseModel
from app.core.schemas import Sample
from app.utils.scorer import EvaluationUtils
import logging

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Result of deduplication/filtering operation"""
    accepted: List[Sample]
    rejected: List[Tuple[Sample, str]]  # (sample, reason)

    class Config:
        arbitrary_types_allowed = True


class DeduplicationHelper:
    """
    Helper class for handling deduplication and filtering logic for data.

    This helper provides functionality to deduplicate data samples or strings based on ROUGE-L scores
    and filter new data against existing datasets.
    """

    def __init__(self, rouge_threshold: float = 0.6):
        """
        Initialize the deduplication helper.

        Args:
            rouge_threshold: The ROUGE-L threshold for deduplication (default: 0.6)
        """
        self.rouge_threshold = rouge_threshold

    async def deduplicate(self, data: List[Sample]) -> FilterResult:
        """
        Remove duplicates from a list of samples based on ROUGE-L similarity.

        Args:
            data: List of samples to deduplicate

        Returns:
            FilterResult with accepted (unique) and rejected (duplicate) samples
        """
        deduped = {}
        rejected = []

        for item in data:
            # First item is always added
            if len(deduped.keys()) == 0:
                deduped[item.msg] = item
                continue

            # Check against existing items
            is_unique = True
            matched_msg = None
            for existing_item in deduped.values():
                rouge = await EvaluationUtils.score_rouge(
                    ref=existing_item.msg,
                    pred=item.msg,
                    rouge_type="rougeL",
                    mode="precision"
                )

                # If ROUGE score is above threshold, consider it a duplicate
                if rouge >= self.rouge_threshold:
                    is_unique = False
                    matched_msg = existing_item.msg[:50]  # Store first 50 chars for reference
                    break

            if is_unique:
                deduped[item.msg] = item
            else:
                reason = f"internal_duplicate (ROUGE-L {rouge:.2f} with '{matched_msg}...')"
                rejected.append((item, reason))

        logger.debug(f"Deduplicated {len(data)} samples to {len(deduped)} unique samples ({len(rejected)} duplicates)")
        return FilterResult(accepted=list(deduped.values()), rejected=rejected)

    async def filter_against_existing(self, new_data: List[Sample], existing_data: List[Sample], window_size: int = 1000) -> FilterResult:
        """
        Filter new data against existing data to avoid duplicates.
        Only compares samples within the same label and uses a sliding window for better performance.

        Args:
            new_data: New samples to filter
            existing_data: Existing samples to compare against
            window_size: Maximum number of recent samples per label to compare against (default: 1000)

        Returns:
            FilterResult with accepted (unique) and rejected (duplicate) samples
        """
        if not existing_data:
            return FilterResult(accepted=new_data, rejected=[])

        # Group existing data by label for efficient lookup
        existing_by_label = {}
        for item in existing_data:
            label = item.label
            if label not in existing_by_label:
                existing_by_label[label] = []
            existing_by_label[label].append(item)

        # Apply sliding window to each label group (keep only recent samples)
        for label in existing_by_label:
            if len(existing_by_label[label]) > window_size:
                existing_by_label[label] = existing_by_label[label][-window_size:]

        filtered = []
        rejected = []

        for new_item in new_data:
            is_unique = True
            matched_msg = None
            matched_rouge = 0.0

            # Only compare against recent existing items with the same label
            same_label_existing = existing_by_label.get(new_item.label, [])

            for existing_item in same_label_existing:
                rouge = await EvaluationUtils.score_rouge(
                    ref=new_item.msg,
                    pred=existing_item.msg,
                    rouge_type="rougeL",
                    mode="precision"
                )

                # If ROUGE score is above threshold, consider it too similar
                if rouge >= self.rouge_threshold:
                    is_unique = False
                    matched_msg = existing_item.msg[:50]
                    matched_rouge = rouge
                    break

            if is_unique:
                filtered.append(new_item)
            else:
                reason = f"external_duplicate (ROUGE-L {matched_rouge:.2f} with '{matched_msg}...')"
                rejected.append((new_item, reason))

        total_comparisons_before = len(new_data) * len(existing_data) if existing_data else 0
        total_comparisons_after = sum(len(existing_by_label.get(item.label, [])) for item in new_data)

        logger.debug(f"Filtered {len(new_data)} new samples to {len(filtered)} unique samples ({len(rejected)} duplicates)")
        logger.debug(f"Label-based + sliding window filtering reduced comparisons from {total_comparisons_before} to {total_comparisons_after}")
        logger.debug(f"Window size: {window_size} per label")
        return FilterResult(accepted=filtered, rejected=rejected)
