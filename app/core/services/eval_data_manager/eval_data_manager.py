from typing import List, Type
import json
import logging
import os
import re
from datasets import Dataset
from app.core.schemas.workflow import Sample, BaseLabelConfig
from app.core.mixins import NumericalFileAccessHelper, DeduplicationHelper

logger = logging.getLogger(__name__)

class EvalDataManager:
    def __init__(self, label_config: Type[BaseLabelConfig], rouge_threshold: float = 0.6, base_dir: str = './cache/eval'):
        self.label_config = label_config

        # Create label-specific eval path
        label_name = self._sanitize_name(label_config.name())

        os.makedirs(f'{base_dir}/{label_name}/', exist_ok=True)
        self._file_helper = NumericalFileAccessHelper(f'{base_dir}/{label_name}/')
        self._dedup_helper = DeduplicationHelper(rouge_threshold)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize label config name for use in file paths"""
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    def save(self, data: List[Sample], iteration_number: int = None):
        """
        Save evaluation data to a numbered folder.

        Args:
            data: List of samples to save
            iteration_number: Optional iteration number. If None, uses next available number.
        """
        if iteration_number is None:
            iteration_number = self._file_helper._get_next_number()

        self._save(data, iteration_number=iteration_number)
        return iteration_number

    def _save(self, data: List[Sample], iteration_number: int):
        """Internal save method."""
        # Ensure base directory exists
        self._file_helper._ensure_base_directory()

        # Create iteration folder
        iteration_path = self._file_helper._get_item_path(iteration_number)
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
            iteration_path = self._file_helper.get_latest_item_path()
            if iteration_path is None:
                return []
        else:
            iteration_path = self._file_helper.get_item_path_by_number(iteration_number)
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
        return await self._dedup_helper.filter_against_existing(data, existing_data)

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
            iteration_number = self._file_helper._get_next_number()

        self._save_open_intent(data, iteration_number=iteration_number)
        return iteration_number

    def _save_open_intent(self, data: List[str], iteration_number: int):
        """Internal save method for open intent data."""
        # Ensure base directory exists
        self._file_helper._ensure_base_directory()

        # Create iteration folder
        iteration_path = self._file_helper._get_item_path(iteration_number)
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
            iteration_path = self._file_helper.get_latest_item_path()
            if iteration_path is None:
                return []
        else:
            iteration_path = self._file_helper.get_item_path_by_number(iteration_number)
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

        # Convert strings to Sample objects with temp label
        temp_label = "open_intent"
        new_samples = [Sample(msg=msg, label=temp_label) for msg in data]
        existing_samples = [Sample(msg=msg, label=temp_label) for msg in existing_data]

        # Use existing filter method
        filtered_samples = await self._dedup_helper.filter_against_existing(new_samples, existing_samples)

        # Convert back to strings
        return [sample.msg for sample in filtered_samples]

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
        return self._file_helper._get_latest_number()
