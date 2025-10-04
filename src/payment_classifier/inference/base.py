from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datasets import Dataset


class BaseInferencer(ABC):
    """Base class for model inference implementations."""

    def __init__(self, peft_path: str):
        """Initialize the inferencer with a model path.

        Args:
            peft_path: Path to the trained model
        """
        self.peft_path = peft_path

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dict containing model information like labels, radii, etc.
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions on a list of texts.

        Args:
            texts: List of text strings to predict

        Returns:
            List of prediction dictionaries containing label, probability, etc.
        """
        pass

    @abstractmethod
    def evaluate(self, known_intent_data: Dataset, unknown_intent_texts: List[str]) -> Dict[str, Any]:
        """Evaluate the model on known and unknown intent data.

        Args:
            known_intent_data: Dataset with known intents
            unknown_intent_texts: List of texts representing unknown intents

        Returns:
            Dict containing comprehensive evaluation metrics
        """
        pass

    @abstractmethod
    def post_train(self, data: Dataset) -> tuple:
        """Post-training processing to calculate model parameters.

        Args:
            data: Training dataset

        Returns:
            Tuple containing calculated parameters (implementation specific)
        """
        pass