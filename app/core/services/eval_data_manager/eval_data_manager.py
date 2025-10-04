from typing import List, Type
import json
import logging
import os
import re
from datasets import Dataset
from app.core.schemas.workflow import Sample, BaseLabelConfig
from app.core.mixins import NumericalFileAccessMixin, DeduplicationMixin

logger = logging.getLogger(__name__)

class EvalDataManager(NumericalFileAccessMixin, DeduplicationMixin):
    def __init__(self, label_config: Type[BaseLabelConfig], rouge_threshold: float = 0.6):
        self._rouge_threshold = rouge_threshold
        self.label_config = label_config

        # Create label-specific eval path
        label_name = self._sanitize_name(label_config.name())
        self.BASE_LOCAL_EVAL_PATH = f'.cache/eval/{label_name}/'

        # Ensure directory exists
        os.makedirs(self.BASE_LOCAL_EVAL_PATH, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize label config name for use in file paths"""
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    @property
    def base_directory(self) -> str:
        return self.BASE_LOCAL_EVAL_PATH

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

        # Save data as JSONL with integer labels
        eval_file = iteration_path / "eval_data.jsonl"
        with open(eval_file, 'w') as f:
            for sample in data:
                eval_sample = {
                    "msg": sample.msg,
                    "label": self.label_config.from_str(sample.label)
                }
                f.write(json.dumps(eval_sample) + '\n')

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
                # Convert integer label back to string for Sample object
                sample_data = {
                    "msg": data["msg"],
                    "label": self.label_config.to_str(data["label"])
                }
                samples.append(Sample(**sample_data))

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

    def save_open_intent(self, data: List[str], iteration_number: int = None):
        """
        Save open intent evaluation data to a numbered folder.

        Args:
            data: List of open intent messages to save
            iteration_number: Optional iteration number. If None, uses next available number.
        """
        if iteration_number is None:
            iteration_number = self._get_next_number()

        self._save_open_intent(data, iteration_number=iteration_number)
        return iteration_number

    def _save_open_intent(self, data: List[str], iteration_number: int):
        """Internal save method for open intent data."""
        # Ensure base directory exists
        self._ensure_base_directory()

        # Create iteration folder
        iteration_path = self._get_item_path(iteration_number)
        iteration_path.mkdir(exist_ok=True)

        # Save data as simple text file, one message per line
        eval_file = iteration_path / "open_intent_data.txt"
        with open(eval_file, 'w') as f:
            for message in data:
                f.write(message + '\n')

        logger.info(f"Saved {len(data)} open intent evaluation messages to {eval_file}")

    def load_open_intent(self, iteration_number: int = None) -> List[str]:
        """
        Load open intent evaluation data from a specific iteration.

        Args:
            iteration_number: Iteration to load. If None, loads latest.

        Returns:
            List of open intent messages from the iteration
        """
        if iteration_number is None:
            iteration_path = self.get_latest_item_path()
            if iteration_path is None:
                return []
        else:
            iteration_path = self.get_item_path_by_number(iteration_number)
            if iteration_path is None:
                return []

        eval_file = iteration_path / "open_intent_data.txt"
        if not eval_file.exists():
            return []

        messages = []
        with open(eval_file, 'r') as f:
            for line in f:
                message = line.strip()
                if message:
                    messages.append(message)

        return messages

    async def filter_open_intent(self, data: List[str], iteration_number: int = None) -> List[str]:
        """
        Filter new open intent data against existing open intent data in the specified iteration.

        Args:
            data: New open intent messages to filter
            iteration_number: Iteration to compare against. If None, uses latest.

        Returns:
            List of filtered open intent messages
        """
        existing_data = self.load_open_intent(iteration_number)
        return await self._filter_strings_against_existing(data, existing_data)

    def append_open_intent(self, data: List[str], iteration_number: int):
        """
        Append new open intent data to an existing iteration.

        Args:
            data: List of open intent messages to append
            iteration_number: Iteration number to append to
        """
        # Load existing data
        existing_data = self.load_open_intent(iteration_number)

        # Combine with new data
        all_data = existing_data + data

        # Save combined data
        self._save_open_intent(all_data, iteration_number)

    async def deduplicate_open_intent(self, data: List[str]) -> List[str]:
        """
        Deduplicate open intent messages using ROUGE-L similarity.

        Args:
            data: List of messages to deduplicate

        Returns:
            Deduplicated list of messages
        """
        return await self._deduplicate_strings(data)

    async def _filter_strings_against_existing(self, new_data: List[str], existing_data: List[str]) -> List[str]:
        """Filter string messages against existing string messages using ROUGE-L."""
        if not existing_data:
            return new_data

        filtered_data = []
        for new_msg in new_data:
            is_similar = False
            for existing_msg in existing_data:
                similarity = await self._calculate_rouge_l_similarity(new_msg, existing_msg)
                if similarity > self.rouge_threshold:
                    is_similar = True
                    break

            if not is_similar:
                filtered_data.append(new_msg)

        return filtered_data

    async def _deduplicate_strings(self, data: List[str]) -> List[str]:
        """Deduplicate a list of string messages using ROUGE-L similarity."""
        if not data:
            return []

        deduplicated = [data[0]]  # Always keep the first item

        for item in data[1:]:
            is_similar = False
            for existing_item in deduplicated:
                similarity = await self._calculate_rouge_l_similarity(item, existing_item)
                if similarity > self.rouge_threshold:
                    is_similar = True
                    break

            if not is_similar:
                deduplicated.append(item)

        return deduplicated

    async def _calculate_rouge_l_similarity(self, text1: str, text2: str) -> float:
        """Calculate ROUGE-L similarity between two strings."""
        from app.utils.scorer import EvaluationUtils
        return await EvaluationUtils.score_rouge(text1, text2, rouge_type="rougeL")

    def to_datasets(self, iteration_number: int = None) -> Dataset:
        """
        Convert evaluation data to HuggingFace datasets format.

        Args:
            iteration_number: Iteration to load. If None, loads latest.

        Returns:
            Dataset: The converted dataset
        """
        # Load all samples from the specified iteration
        samples = self.load(iteration_number)
        transformed = [{
            "msg": sample.msg,
            "label": self.label_config.from_str(sample.label)
        } for sample in samples]

        if not samples:
            # Create empty dataset if no data
            data_dict = {"msg": [], "label": []}
        else:
            # Convert samples to dataset format
            data_dict = {
                "msg": [sample['msg'] for sample in transformed],
                "label": [sample['label'] for sample in transformed]
            }

        # Create dataset
        dataset = Dataset.from_dict(data_dict)
        return dataset

    def get_latest_item_number(self) -> int:
        """
        Get the latest (highest) iteration number.

        Returns:
            int: Latest iteration number
        """
        return self._get_latest_number()
