from typing import List
from datasets import Dataset
from app.core.schemas import Sample
from app.core.models.models import LabelConfig
from app.core.mixins import DeduplicationHelper
import json
import os
import re

class DataManager:
    def __init__(
            self,
            label_config: LabelConfig,
            rouge_threshold: float = 0.6,
            base_dir: str = '.cache',
            absolute_dir: str | None = None
        ):
        self.label_config = label_config
        self.cache_dir = base_dir
        self.LOCAL_FILE = f'{self.cache_dir}/.data.jsonl'
        if absolute_dir:
            self.cache_dir = absolute_dir.replace('/.data.jsonl', '')
            self.LOCAL_FILE = absolute_dir
        else:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Ensure directory exists

        # Initialize deduplication helper
        self._dedup_helper = DeduplicationHelper(rouge_threshold)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize label config name for use in file paths"""
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    def save(self, data: List[Sample]):
        """Append data to the base file."""
        self._save(data, path=self.LOCAL_FILE, mode='a')

    def hard_save(self, data: List[Sample]):
        """Overwrite the base file with new data (does not append)."""
        self._save(data, path=self.LOCAL_FILE, mode='w')

    def save_to_versioned_file(self, data: List[Sample], file_suffix: str):
        """Save data to a versioned file with a custom suffix.

        Args:
            data: List of samples to save
            file_suffix: Suffix to append to the file name (e.g., 'fix_gen_1')
        """
        versioned_file = f'{self.cache_dir}/.data_{file_suffix}.jsonl'
        self._save(data, path=versioned_file, mode='w')
        return versioned_file

    def _save(self, data: List[Sample], path: str = None, mode: str = 'a'):
        """Internal save method.

        Args:
            data: List of samples to save
            path: Path to save to
            mode: File open mode ('a' for append, 'w' for overwrite)
        """
        label2id = self.label_config.get_label2id()
        with open(path, mode) as f:
            for item in data:
                data_dict = item.model_dump()
                data_dict['label'] = label2id[item.label]
                s = json.dumps(data_dict)
                f.write(f"{s}\n")


    async def filter(self, data: List[Sample]):
        """Filter new data against existing data in the local file.

        Returns:
            FilterResult with accepted and rejected samples
        """
        from app.core.mixins.deduplication import FilterResult
        existing_data = self.load()
        return await self._dedup_helper.filter_against_existing(data, existing_data)
    
    def load(self) -> List[Sample]:
        # Load data from file
        loaded_data = []
        if not os.path.exists(self.LOCAL_FILE):
            return loaded_data

        id2label = self.label_config.get_id2label()
        with open(self.LOCAL_FILE, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['label'] = id2label[str(item['label'])]
                loaded_data.append(Sample.model_validate(item))
        return loaded_data

    def load_from_path(self, file_path: str) -> List[Sample]:
        """Load data from a specific file path.

        Args:
            file_path: Path to the JSONL file to load

        Returns:
            List of Sample objects
        """
        loaded_data = []
        if not os.path.exists(file_path):
            return loaded_data

        id2label = self.label_config.get_id2label()
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['label'] = id2label[str(item['label'])]
                loaded_data.append(Sample.model_validate(item))
        return loaded_data

    def to_datasets(self, file_path: str = None) -> Dataset:
        """
        Convert data to HuggingFace datasets format

        Args:
            file_path: Optional path to specific JSONL file. If None, uses LOCAL_FILE

        Returns:
            Dataset: The converted dataset
        """
        # Load samples from specified path or default LOCAL_FILE
        if file_path:
            samples = self.load_from_path(file_path)
        else:
            samples = self.load()

        label2id = self.label_config.get_label2id()
        transformed = [{
            "msg": sample.msg,
            "label": label2id[sample.label]
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

    def combine_versioned_files_to_dataset(self, versioned_files: List[str]) -> Dataset:
        """Combine multiple versioned files into a single dataset

        Args:
            versioned_files: List of paths to versioned JSONL files

        Returns:
            Combined HuggingFace Dataset
        """
        import logging
        logger = logging.getLogger(__name__)

        all_samples = []

        for file_path in versioned_files:
            try:
                samples = self.load_from_path(file_path)
                all_samples.extend(samples)
                logger.debug(f"Loaded {len(samples)} samples from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load from {file_path}: {e}")

        # Convert to dataset format
        if not all_samples:
            return Dataset.from_dict({"msg": [], "label": []})

        label2id = self.label_config.get_label2id()
        data_dict = {
            "msg": [sample.msg for sample in all_samples],
            "label": [label2id[sample.label] for sample in all_samples]
        }

        return Dataset.from_dict(data_dict)

    def get_label_counts(self, file_path: str = None) -> dict:
        """Get the count of samples for each label.

        Args:
            file_path: Optional path to specific JSONL file. If None, uses LOCAL_FILE

        Returns:
            Dictionary mapping label strings to their counts
        """
        # Load samples from specified path or default LOCAL_FILE
        if file_path:
            samples = self.load_from_path(file_path)
        else:
            samples = self.load()

        # Count labels
        label_counts = {}
        for sample in samples:
            label = sample.label
            label_counts[label] = label_counts.get(label, 0) + 1

        return label_counts