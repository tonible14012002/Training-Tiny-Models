from typing import List, Optional
from datasets import Dataset
from app.core.schemas import Sample, PAYMENT_LABEL
from app.utils.scorer import EvaluationUtils
import json

class DataManager:
    LOCAL_FILE = '.cache/.data.jsonl'

    def __init__(
            self,
            rouge_threshold: float = 0.6,
        ):
        self.ROUGE_THRESHOLD = rouge_threshold

    def save(self, data: List[Sample]):
        # open File and append data
        with open(self.LOCAL_FILE, 'a') as f:
            for item in data:
                data_dict = item.model_dump()
                data_dict['label'] = PAYMENT_LABEL.from_str(item.label)
                s = json.dumps(data_dict)
                f.write(f"{s}\n")

    async def deduplicate(self, data: List[Sample]) -> List[Sample]:
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
    
    async def filter(self, data: List[Sample]) -> List[Sample]:
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