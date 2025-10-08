from typing import List
from datasets import Dataset
from app.core.schemas.workflow import Sample
from app.core.models.models import LabelConfig


class DatasetHelper:
    """Helper class for Dataset operations"""

    @staticmethod
    def ds_to_samples(
        dataset: Dataset,
        label_config: LabelConfig
    ) -> List[Sample]:
        """
        Convert a dataset with integer labels to a list of Sample instances with string labels.

        Args:
            dataset: HuggingFace Dataset instance with integer labels
            label_config: LabelConfig model instance from database

        Returns:
            List[Sample]: List of Sample instances with labels converted from int to string
        """
        samples = []
        id2label = label_config.get_id2label()

        for item in dataset:
            # Extract message and label from dataset item
            msg = item.get('msg') or item.get('text') or item.get('message', '')
            label_int = item.get('label')

            # Convert integer label to string using id2label dict
            label_str = id2label[str(label_int)]

            # Create Sample instance
            sample = Sample(msg=msg, label=label_str)
            samples.append(sample)

        return samples

    @staticmethod
    def json_to_ds(
        json_list: List[dict],
        label_config: LabelConfig
    ) -> Dataset:

        data = {
            'msg': [],
            'label': []
        }

        for item in json_list:
            data['msg'].append(item.get('msg', ''))
            label_int = item.get('label')
            data['label'].append(label_int)

        return Dataset.from_dict(data)