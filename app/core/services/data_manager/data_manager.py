from typing import List, Optional
import os
from datasets import Dataset
from app.core.schemas import Sample
from app.utils.scorer import EvaluationUtils

class DataManager:
    def __init__(
            self,
            rouge_threshold: float = 0.6,
        ):
        self.ROUGE_THRESHOLD = rouge_threshold

    def save(self, data: List[Sample]):
        # open File and append data
        with open('data.txt', 'a') as f:
            for item in data:
                f.write(f"{item.model_dump_json()}\n")

    async def _deduplicate(self, data: List[Sample]) -> List[Sample]:
        deduped = {}
        for item in data:
            pass
            # Filter low ROUGE-L score
            if len(deduped.keys()) == 0:
                deduped[item.msg] = item
                continue
            
            ok = True
            for existing_item in deduped.values():
                rouge = await EvaluationUtils.score_rouge(
                    ref=existing_item.msg,
                    pred=item.msg,
                    rouge_type="rougeL",
                    mode="precision"
                )

                if rouge < self.ROUGE_THRESHOLD:
                    ok = False
                    break
            
            if ok:
                deduped[item.msg] = item

        return list(deduped.values())
    
    async def filter(self, data: List) -> List:
        existed_data = self.load()

        add = []
        # Filter low ROUGE-L score against existed data
        for item in data:
            for existing_item in existed_data:
                rouge = await EvaluationUtils.score_rouge(
                    ref=item.msg,
                    pred=existing_item.msg,
                    rouge_type="rougeL",
                    mode="precision"
                )
                if rouge < self.ROUGE_THRESHOLD:
                    add.append(item)
                    break
                
        return add
    
    def load(self) -> List[Sample]:
        # Load data from file
        loaded_data = []
        with open('data.txt', 'r') as f:
            for line in f:
                loaded_data.append(Sample.model_validate_json(line.strip()))
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

        if not samples:
            # Create empty dataset if no data
            data_dict = {"msg": [], "label": []}
        else:
            # Convert samples to dataset format
            data_dict = {
                "msg": [sample.msg for sample in samples],
                "label": [sample.label for sample in samples]
            }

        # Create dataset
        dataset = Dataset.from_dict(data_dict)
        return dataset