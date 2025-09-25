from typing import List
from app.core.schemas import Sample
from app.utils.scorer import EvaluationUtils
import logging

logger = logging.getLogger(__name__)


class DeduplicationMixin:
    """
    Mixin class for handling deduplication and filtering logic for data.

    This mixin provides functionality to deduplicate data samples based on ROUGE-L scores
    and filter new data against existing datasets.

    Classes using this mixin should define:
    - rouge_threshold: float - The ROUGE-L threshold for deduplication
    """

    @property
    def rouge_threshold(self) -> float:
        """
        ROUGE-L threshold for deduplication.
        Must be implemented by the class using this mixin.
        """
        raise NotImplementedError("Classes using DeduplicationMixin must define rouge_threshold")

    async def deduplicate(self, data: List[Sample]) -> List[Sample]:
        """
        Remove duplicates from a list of samples based on ROUGE-L similarity.

        Args:
            data: List of samples to deduplicate

        Returns:
            List of deduplicated samples
        """
        deduped = {}

        for item in data:
            # First item is always added
            if len(deduped.keys()) == 0:
                deduped[item.msg] = item
                continue

            # Check against existing items
            is_unique = True
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
                    break

            if is_unique:
                deduped[item.msg] = item

        logger.debug(f"Deduplicated {len(data)} samples to {len(deduped)} unique samples")
        return list(deduped.values())

    async def filter_against_existing(self, new_data: List[Sample], existing_data: List[Sample]) -> List[Sample]:
        """
        Filter new data against existing data to avoid duplicates.

        Args:
            new_data: New samples to filter
            existing_data: Existing samples to compare against

        Returns:
            List of filtered samples that are sufficiently different from existing data
        """
        if not existing_data:
            return new_data

        filtered = []

        for new_item in new_data:
            is_unique = True

            for existing_item in existing_data:
                rouge = await EvaluationUtils.score_rouge(
                    ref=new_item.msg,
                    pred=existing_item.msg,
                    rouge_type="rougeL",
                    mode="precision"
                )

                # If ROUGE score is above threshold, consider it too similar
                if rouge >= self.rouge_threshold:
                    is_unique = False
                    break

            if is_unique:
                filtered.append(new_item)

        logger.debug(f"Filtered {len(new_data)} new samples to {len(filtered)} unique samples")
        return filtered

    async def deduplicate_and_filter(self, new_data: List[Sample], existing_data: List[Sample] = None) -> List[Sample]:
        """
        Combined deduplication and filtering operation.

        Args:
            new_data: New samples to process
            existing_data: Optional existing samples to filter against

        Returns:
            List of processed samples
        """
        # First deduplicate within new data
        deduped_data = await self.deduplicate(new_data)

        # Then filter against existing data if provided
        if existing_data:
            filtered_data = await self.filter_against_existing(deduped_data, existing_data)
            return filtered_data

        return deduped_data