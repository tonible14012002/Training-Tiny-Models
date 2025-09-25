from typing import List, Optional
from datasets import Dataset
from app.core.schemas import Sample, PAYMENT_LABEL
from app.core.mixins import DeduplicationMixin
import json

class DataManager(DeduplicationMixin):
    LOCAL_FILE = '.cache/.data.jsonl'

    def __init__(
            self,
            rouge_threshold: float = 0.6,
        ):
        self._rouge_threshold = rouge_threshold

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
                data_dict['label'] = PAYMENT_LABEL.from_str(item.label)
                s = json.dumps(data_dict)
                f.write(f"{s}\n")

    
    async def filter(self, data: List[Sample]) -> List[Sample]:
        """Filter new data against existing data in the local file."""
        existing_data = self.load()
        return await self.filter_against_existing(data, existing_data)
    
    def load(self) -> List[Sample]:
        # Load data from file
        loaded_data = []
        with open(self.LOCAL_FILE, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['label'] = PAYMENT_LABEL.to_str(item['label'])
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
            "label": PAYMENT_LABEL.from_str(sample.label)
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