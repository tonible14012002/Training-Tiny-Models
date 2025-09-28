from typing import List, Optional, Type
from datasets import Dataset
from app.core.schemas import Sample
from app.core.schemas.workflow import BaseLabelConfig
from app.core.mixins import DeduplicationMixin
import json
import os
import re

class DataManager(DeduplicationMixin):
    def __init__(
            self,
            label_config: Type[BaseLabelConfig],
            rouge_threshold: float = 0.6
        ):
        self._rouge_threshold = rouge_threshold
        self.label_config = label_config

        # Create label-specific file path
        label_name = self._sanitize_name(label_config.name())
        self.cache_dir = f'.cache/{label_name}'
        self.LOCAL_FILE = f'{self.cache_dir}/.data.jsonl'

        # Ensure directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize label config name for use in file paths"""
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    @property
    def rouge_threshold(self) -> float:
        return self._rouge_threshold

    def save(self, data: List[Sample]):
        self._save(data, path=self.LOCAL_FILE)

    def _save(self, data: List[Sample], path: str = None):
        # open File and append data
        with open(path, 'a') as f:
            for item in data:
                data_dict = item.model_dump()
                data_dict['label'] = self.label_config.from_str(item.label)
                s = json.dumps(data_dict)
                f.write(f"{s}\n")

    
    async def filter(self, data: List[Sample]) -> List[Sample]:
        """Filter new data against existing data in the local file."""
        existing_data = self.load()
        return await self.filter_against_existing(data, existing_data)
    
    def load(self) -> List[Sample]:
        # Load data from file
        loaded_data = []
        if not os.path.exists(self.LOCAL_FILE):
            return loaded_data

        with open(self.LOCAL_FILE, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['label'] = self.label_config.to_str(item['label'])
                loaded_data.append(Sample.model_validate(item))
        return loaded_data

    def to_datasets(self) -> Dataset:
        """
        Convert current data.txt format to HuggingFace datasets format

        Args:
            output_path: Path where to save the dataset

        Returns:
            Dataset: The converted dataset
        """
        # Load all samples from data.txt
        samples = self.load()
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