from typing import List
import json
import logging
from app.core.schemas.workflow import Sample
from app.core.mixins import NumericalFileAccessMixin, DeduplicationMixin

logger = logging.getLogger(__name__)

class EvalDataManager(NumericalFileAccessMixin, DeduplicationMixin):
    BASE_LOCAL_EVAL_PATH = '.cache/eval/'

    @property
    def base_directory(self) -> str:
        return self.BASE_LOCAL_EVAL_PATH

    def __init__(self, rouge_threshold: float = 0.6):
        self._rouge_threshold = rouge_threshold

    @property
    def rouge_threshold(self) -> float:
        return self._rouge_threshold

    def save(self, data: List[Sample], iteration_number: int = None):
        """
        Save evaluation data to a numbered folder.

        Args:
            data: List of samples to save
            iteration_number: Optional iteration number. If None, uses next available number.
        """
        if iteration_number is None:
            iteration_number = self._get_next_number()

        self._save(data, iteration_number=iteration_number)
        return iteration_number

    def _save(self, data: List[Sample], iteration_number: int):
        """Internal save method."""
        # Ensure base directory exists
        self._ensure_base_directory()

        # Create iteration folder
        iteration_path = self._get_item_path(iteration_number)
        iteration_path.mkdir(exist_ok=True)

        # Save data as JSONL
        eval_file = iteration_path / "eval_data.jsonl"
        with open(eval_file, 'w') as f:
            for sample in data:
                f.write(json.dumps(sample.model_dump()) + '\n')

        logger.info(f"Saved {len(data)} evaluation samples to {eval_file}")

    def load(self, iteration_number: int = None) -> List[Sample]:
        """
        Load evaluation data from a specific iteration.

        Args:
            iteration_number: Iteration to load. If None, loads latest.

        Returns:
            List of samples from the iteration
        """
        if iteration_number is None:
            iteration_path = self.get_latest_item_path()
            if iteration_path is None:
                return []
        else:
            iteration_path = self.get_item_path_by_number(iteration_number)
            if iteration_path is None:
                return []

        eval_file = iteration_path / "eval_data.jsonl"
        if not eval_file.exists():
            return []

        samples = []
        with open(eval_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(Sample(**data))

        return samples

    async def filter(self, data: List[Sample], iteration_number: int = None) -> List[Sample]:
        """
        Filter new data against existing data in the specified iteration.

        Args:
            data: New samples to filter
            iteration_number: Iteration to compare against. If None, uses latest.

        Returns:
            List of filtered samples
        """
        existing_data = self.load(iteration_number)
        return await self.filter_against_existing(data, existing_data)

    def append(self, data: List[Sample], iteration_number: int):
        """
        Append new data to an existing iteration.

        Args:
            data: List of samples to append
            iteration_number: Iteration number to append to
        """
        # Load existing data
        existing_data = self.load(iteration_number)

        # Combine with new data
        all_data = existing_data + data

        # Save combined data
        self._save(all_data, iteration_number)

