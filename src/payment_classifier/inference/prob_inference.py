from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
from app.core.schemas.workflow import BaseLabelConfig
from datasets import Dataset
import torch
from typing import List, Type, Dict, Any
from collections import defaultdict


class ProbModelInference:
    def __init__(
        self,
        peft_path: str,
        label_config: Type[BaseLabelConfig]
    ):
        self.peft_path = peft_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_config = label_config
        self._setup(peft_path)

    def info(self):
        return {
            "labels": self.id2label,
            "inference_type": "probability_based",
        }

    def _setup(self, peft_path: str, tokenizer=None):
        self.peft_path = peft_path
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(peft_path)

        # Use injected label configuration
        self.label2id = self.label_config.to_dict()
        self.id2label = self.label_config.to_id2label()

        self.peft_model = AutoPeftModelForSequenceClassification.from_pretrained(
            peft_path,
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.peft_model.to(self.device)

    def post_train(self, data: Dataset):
        """For compatibility with the base interface. No post-training needed for probability-based inference."""
        return {}, {}

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict using raw model probabilities without ADB filtering."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.peft_model(**inputs)
            probs = outputs.logits.softmax(dim=-1)

        predictions = []
        for i in range(len(texts)):
            # Get the predicted label (highest probability)
            predicted_label_id = probs[i].argmax().item()
            predicted_prob = probs[i, predicted_label_id].item()

            # Get all probabilities for this sample
            all_probs = {
                self.id2label[label_id]: probs[i, label_id].item()
                for label_id in range(len(self.id2label))
            }

            predicted_label = {
                "label": self.id2label[predicted_label_id],
                "prob": predicted_prob,
                "all_probs": all_probs,
            }

            predictions.append(predicted_label)

        return predictions

    def evaluate(self, known_intent_data: Dataset, unknown_intent_texts: List[str]) -> dict:
        """
        Evaluate using probability-based predictions.
        Note: This doesn't handle unknown intents as there's no ADB mechanism.
        All predictions will be mapped to known labels.
        """
        # Get predictions for known intents
        known_predictions = []
        known_true_labels = []

        batch_size = 32
        for i in range(0, len(known_intent_data), batch_size):
            batch = known_intent_data[i:i+batch_size]
            texts = batch['msg']
            true_labels = batch['label']

            predictions = self.predict(texts)
            known_predictions.extend(predictions)
            known_true_labels.extend(true_labels)

        # For unknown intents, we still predict but note they'll be mapped to known labels
        unknown_predictions = []
        if unknown_intent_texts:
            for i in range(0, len(unknown_intent_texts), batch_size):
                batch_texts = unknown_intent_texts[i:i+batch_size]
                predictions = self.predict(batch_texts)
                unknown_predictions.extend(predictions)

        # Calculate metrics for known intents only (probability-based doesn't handle unknowns)
        all_predictions = known_predictions
        all_true_labels = [self.id2label[label] for label in known_true_labels]

        total_samples = len(all_predictions)
        all_labels = set(self.id2label.values())

        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        true_negatives = defaultdict(int)
        false_negatives = defaultdict(int)

        correct = 0

        # Calculate metrics
        for pred, true_label_str in zip(all_predictions, all_true_labels):
            pred_label = pred["label"]

            # Overall accuracy
            if pred_label == true_label_str:
                correct += 1

            # Binary classification metrics for each label
            for label in all_labels:
                true_is_label = (true_label_str == label)
                pred_is_label = (pred_label == label)

                if true_is_label and pred_is_label:
                    true_positives[label] += 1
                elif not true_is_label and pred_is_label:
                    false_positives[label] += 1
                elif true_is_label and not pred_is_label:
                    false_negatives[label] += 1
                else:
                    true_negatives[label] += 1

        # Calculate per-label metrics
        per_label_metrics = {}
        macro_precision_sum = 0
        macro_recall_sum = 0
        macro_f1_sum = 0
        valid_labels = 0

        for label in all_labels:
            tp = true_positives[label]
            fp = false_positives[label]
            tn = true_negatives[label]
            fn = false_negatives[label]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / total_samples

            actual_samples = tp + fn

            per_label_metrics[label] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "samples": actual_samples,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }

            # Include in macro averages if the label exists in the dataset
            if actual_samples > 0:
                macro_precision_sum += precision
                macro_recall_sum += recall
                macro_f1_sum += f1
                valid_labels += 1

        # Calculate macro averages
        macro_precision = macro_precision_sum / valid_labels if valid_labels > 0 else 0.0
        macro_recall = macro_recall_sum / valid_labels if valid_labels > 0 else 0.0
        macro_f1 = macro_f1_sum / valid_labels if valid_labels > 0 else 0.0

        overall_accuracy = correct / total_samples if total_samples > 0 else 0.0

        return {
            "overall": {
                "accuracy": overall_accuracy,
                "unknown_rate": 0.0,  # No unknown handling in probability-based inference
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "total_samples": total_samples,
                "correct_predictions": correct,
                "unknown_predictions": 0,  # No unknown predictions
                "known_intent_samples": len(known_intent_data),
                "unknown_intent_samples": len(unknown_intent_texts) if unknown_intent_texts else 0
            },
            "per_label": per_label_metrics,
            "unknown_analysis": {
                "note": "Probability-based inference does not support unknown intent detection",
                "unknown_predictions_mapped_to_known": len(unknown_predictions) if unknown_predictions else 0
            },
            "inference_info": {
                "type": "probability_based",
                "labels": self.id2label
            }
        }