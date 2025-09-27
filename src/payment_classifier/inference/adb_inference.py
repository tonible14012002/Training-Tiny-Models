from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
from app.core.schemas import PAYMENT_LABEL
from datasets import Dataset
import torch
from typing import List
from collections import defaultdict
import json
from pathlib import Path

class ADBModelInference:
    def __init__(
        self,
        peft_path: str,
    ):
        self.peft_path = peft_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup(peft_path)
        self._load_adb_data(peft_path)

    def info(self):
        return {
            "labels": self.id2label,
            "radii": self.intent_radii if self.intent_radii else None,
        }

    def _setup(self, peft_path: str, tokenizer=None):
        self.peft_path = peft_path
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(peft_path)

        # Use centralized label configuration
        self.label2id = PAYMENT_LABEL.to_dict()
        self.id2label = PAYMENT_LABEL.to_id2label()

        self.peft_model = AutoPeftModelForSequenceClassification.from_pretrained(
            peft_path,
            label2id=self.label2id,
            id2label=self.id2label,
        )
        # Enable show hidden states for debugging
        self.peft_model.config.output_hidden_states = True

    def _load_adb_data(self, peft_path: str):
        """Load ADB centers and radii from saved data, if available"""
        adb_file_path = Path(peft_path) / "adb_data.json"

        if adb_file_path.exists():
            with open(adb_file_path, 'r') as f:
                adb_data = json.load(f)

            # Convert centers back to tensors
            self.intent_centers = {
                int(label): torch.tensor(center, device=self.device)
                for label, center in adb_data["intent_centers"].items()
            }
            self.intent_radii = {
                int(label): radius
                for label, radius in adb_data["intent_radii"].items()
            }
        else:
            # ADB data not found - will be loaded when needed
            self.intent_centers = None
            self.intent_radii = None

    def check_adb(self):
        """Check if ADB data is available, raise assertion error if not"""
        adb_file_path = Path(self.peft_path) / "adb_data.json"
        assert adb_file_path.exists(), (
            f"ADB data not found at {adb_file_path}. "
            "Please run training first to calculate ADB centers and radii, "
            "or manually call calc_adb() and save the data."
        )

    def predict(self, texts: list[str], return_cls_emb: bool) -> list[int]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        outputs = self.peft_model(**inputs, output_hidden_states=True)
        predictions = outputs.logits.argmax(dim=-1).tolist()

        # Convert logits indices to actual label IDs
        sorted_ids = sorted(self.label2id.values())  # [1, 2, 3]
        predicted_label_ids = [sorted_ids[idx] for idx in predictions]

        # Then get the string labels
        predicted_labels = [self.id2label[label_id] for label_id in predicted_label_ids]

        if return_cls_emb:
            return predicted_labels, outputs, outputs.hidden_states[-1][:, 0, :]  # CLS token

        return predicted_labels, outputs

    def calc_adb(self, data: Dataset):
        confidence_ratio = 0.9  # For radius calculation
        batch_size = 32
        # Get the embedding dimension from the model
        embedding_dim = self.peft_model.config.hidden_size

        # Running statistics for each label
        label_sums = defaultdict(lambda: torch.zeros(embedding_dim, device=self.device))
        label_counts = defaultdict(int)
        label_embeddings = defaultdict(list)  # Store all embeddings for radius calculation

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                texts = batch['msg']
                labels = batch['label']

                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)

                outputs = self.peft_model(**inputs, output_hidden_states=True)
                cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # CLS token

                for emb, label in zip(cls_embeddings, labels):
                    label_sums[label] += emb
                    label_counts[label] += 1
                    label_embeddings[label].append(emb.cpu())  # Store on CPU to save GPU memory

        # Calculate centers
        intent_centers = {}
        for label in label_sums.keys():
            intent_centers[label] = label_sums[label] / label_counts[label]

        # Calculate optimized radii
        intent_radii = {}
        for label, center in intent_centers.items():
            embeddings_tensor = torch.stack(label_embeddings[label])  # [num_samples, 128]

            # Calculate distances from center to all training samples
            distances = torch.norm(embeddings_tensor - center.unsqueeze(0), dim=1)

            # Use percentile-based radius for tight boundary
            radius = torch.quantile(distances, confidence_ratio)
            intent_radii[label] = radius.item()

        self.intent_centers = intent_centers
        self.intent_radii = intent_radii

        # Convert tensors to lists for JSON serialization
        intent_centers_serializable = {
            label: center.cpu().tolist() for label, center in intent_centers.items()
        }

        return intent_centers_serializable, intent_radii
    
    def predict_with_adb(self, texts: List[str]) -> List[str]:
        self.check_adb()
        assert self.intent_centers is not None and self.intent_radii is not None, (
            "ADB parameters not loaded. Please ensure ADB data was properly calculated and saved."
        )

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.peft_model(**inputs, output_hidden_states=True)
            print(outputs)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # CLS token
            probs = outputs.logits.softmax(dim=-1)

        predictions = []
        for i, emb in enumerate(cls_embeddings):
            detected_intents = []
            min_distance = float('inf')
            closest_intent = None
            for label_id, center in self.intent_centers.items():
                distance = torch.norm(emb - center.to(self.device)).item()

                if distance < self.intent_radii[label_id]:
                    detected_intents.append((label_id, probs[i, label_id].item(), distance))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_intent = label_id
            
            if len(detected_intents) > 0:
                # Get max prob intent among detected intents
                detected_intents.sort(key=lambda x: x[1], reverse=True)
                best_intent = detected_intents[0]
                predicted_label = {
                    "label": self.id2label[best_intent[0]],
                    "prob": best_intent[1],
                    "dis": best_intent[2],
                }
                # Choose the intent with the highest probability among detected intents
            else:
                predicted_label = {
                    "label": "Unknown",
                    "prob": 0.0,
                    "closest": self.id2label[closest_intent],
                    "dis": min_distance,
                }

            predictions.append(predicted_label)

        return predictions

    def evaluate_with_adb(self, data: Dataset) -> dict:
        """
        Evaluate model performance using ADB on a dataset with ground truth labels.

        IMPORTANT: This method evaluates on KNOWN intents only. All samples in the dataset
        are expected to be valid known intents, so they should NOT be predicted as "Unknown".
        The "Unknown" prediction is treated as a classification error (false negative for
        the true label, false positive for Unknown class).

        For testing Unknown/OOD detection capability, use _analyze_open_intents() instead.
        """
        self.check_adb()
        assert self.intent_centers is not None and self.intent_radii is not None, (
            "ADB parameters not loaded. Please ensure ADB data was properly calculated and saved."
        )

        batch_size = 32
        all_predictions = []
        all_true_labels = []

        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            texts = batch['msg']
            true_labels = batch['label']

            # Get predictions for this batch
            predictions = self.predict_with_adb(texts)

            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)

        # Calculate metrics using proper binary classification for each label
        total_samples = len(all_predictions)
        correct = 0
        unknown_predictions = 0

        # Initialize counters for all labels including "Unknown"
        all_labels = set(self.id2label.values()) | {"Unknown"}

        # For each label, treat as binary classification: this_label vs not_this_label
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        true_negatives = defaultdict(int)
        false_negatives = defaultdict(int)


        for pred, true_label in zip(all_predictions, all_true_labels):
            true_label_str = self.id2label[true_label]
            pred_label = pred["label"]

            if pred_label == "Unknown":
                unknown_predictions += 1

            # Overall accuracy: exact match
            if pred_label == true_label_str:
                correct += 1


            # For each possible label, calculate binary classification metrics
            for label in all_labels:
                # True label is this label: positive class
                # True label is not this label: negative class
                true_is_label = (true_label_str == label)
                pred_is_label = (pred_label == label)

                if true_is_label and pred_is_label:
                    true_positives[label] += 1  # Correctly predicted as this label
                elif not true_is_label and pred_is_label:
                    false_positives[label] += 1  # Incorrectly predicted as this label
                elif true_is_label and not pred_is_label:
                    false_negatives[label] += 1  # Should be this label but predicted as something else
                else:  # not true_is_label and not pred_is_label
                    true_negatives[label] += 1  # Correctly predicted as NOT this label

        # Calculate overall metrics
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        unknown_rate = unknown_predictions / total_samples if total_samples > 0 else 0.0

        # Calculate per-label metrics
        per_label_metrics = {}

        macro_precision_sum = 0
        macro_recall_sum = 0
        macro_f1_sum = 0
        macro_accuracy_sum = 0
        valid_labels = 0

        for label in all_labels:
            tp = true_positives[label]
            fp = false_positives[label]
            tn = true_negatives[label]
            fn = false_negatives[label]

            # Every label should be evaluated on the ENTIRE evaluation set
            # So tp + fp + tn + fn should always equal total_samples
            assert tp + fp + tn + fn == total_samples, f"Binary classification counts don't add up for label {label}: {tp + fp + tn + fn} != {total_samples}"

            # Calculate standard binary classification metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            label_accuracy = (tp + tn) / total_samples

            # Count actual samples of this label in the dataset (positive class)
            actual_samples = tp + fn  # True positives + false negatives = all actual instances


            per_label_metrics[label] = {
                "accuracy": label_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "samples": actual_samples,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }

            # Only include labels that actually exist in the dataset for macro averages
            # (i.e., labels that have at least one true positive or false negative)
            if actual_samples > 0:
                macro_precision_sum += precision
                macro_recall_sum += recall
                macro_f1_sum += f1
                macro_accuracy_sum += label_accuracy
                valid_labels += 1

        # Calculate macro averages
        macro_precision = macro_precision_sum / valid_labels if valid_labels > 0 else 0.0
        macro_recall = macro_recall_sum / valid_labels if valid_labels > 0 else 0.0
        macro_f1 = macro_f1_sum / valid_labels if valid_labels > 0 else 0.0
        # macro_accuracy = macro_accuracy_sum / valid_labels if valid_labels > 0 else 0.0

        # Validate known intent true negative requirement:
        # All samples in evaluation dataset are known intents, so they should NOT be predicted as "Unknown"
        known_intent_true_negative_rate = (total_samples - unknown_predictions) / total_samples if total_samples > 0 else 0.0

        return {
            "overall": {
                "accuracy": accuracy,
                "unknown_rate": unknown_rate,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "total_samples": total_samples,
                "correct_predictions": correct,
                "unknown_predictions": unknown_predictions,
                "known_intent_true_negative_rate": known_intent_true_negative_rate
            },
            "per_label": per_label_metrics,
            "adb_info": {
                "radii": self.intent_radii,
                "labels": self.id2label
            }
        }

    def evaluate_with_unknown_intents(self, known_intent_data: Dataset, unknown_intent_texts: List[str]) -> dict:
        """
        Comprehensive evaluation that treats "Unknown" as a proper label to be tested on both:
        1. Known intents (should NOT be predicted as Unknown - true negatives)
        2. Unknown intents (should BE predicted as Unknown - true positives)

        Args:
            known_intent_data: Dataset with known intents (same as regular evaluation dataset)
            unknown_intent_texts: List of texts that represent unknown/out-of-domain intents

        Returns:
            dict: Comprehensive metrics including Unknown label performance
        """
        self.check_adb()
        assert self.intent_centers is not None and self.intent_radii is not None, (
            "ADB parameters not loaded. Please ensure ADB data was properly calculated and saved."
        )

        # Get predictions for known intents
        known_predictions = []
        known_true_labels = []

        batch_size = 32
        for i in range(0, len(known_intent_data), batch_size):
            batch = known_intent_data[i:i+batch_size]
            texts = batch['msg']
            true_labels = batch['label']

            predictions = self.predict_with_adb(texts)
            known_predictions.extend(predictions)
            known_true_labels.extend(true_labels)

        # Get predictions for unknown intents
        unknown_predictions = []
        unknown_true_labels = ["Unknown"] * len(unknown_intent_texts)  # All should be Unknown

        for i in range(0, len(unknown_intent_texts), batch_size):
            batch_texts = unknown_intent_texts[i:i+batch_size]
            predictions = self.predict_with_adb(batch_texts)
            unknown_predictions.extend(predictions)

        # Combine all predictions and true labels
        all_predictions = known_predictions + unknown_predictions
        all_true_labels = [self.id2label[label] for label in known_true_labels] + unknown_true_labels

        total_samples = len(all_predictions)

        # Initialize counters for all labels including "Unknown"
        all_labels = set(self.id2label.values()) | {"Unknown"}

        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        true_negatives = defaultdict(int)
        false_negatives = defaultdict(int)

        correct = 0
        unknown_predictions_count = 0

        # Calculate metrics
        for pred, true_label_str in zip(all_predictions, all_true_labels):
            pred_label = pred["label"]

            if pred_label == "Unknown":
                unknown_predictions_count += 1

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

        # Calculate unknown rate for compatibility with OverallMetrics schema
        # Unknown rate: portion of samples predicted as Unknown
        unknown_rate = unknown_predictions_count / total_samples if total_samples > 0 else 0.0

        # Specific Unknown label analysis
        unknown_tp = true_positives["Unknown"]  # Unknown intents correctly identified as Unknown
        unknown_fp = false_positives["Unknown"]  # Known intents incorrectly identified as Unknown
        unknown_tn = true_negatives["Unknown"]   # Known intents correctly NOT identified as Unknown
        unknown_fn = false_negatives["Unknown"]  # Unknown intents incorrectly identified as known intent

        return {
            "overall": {
                "accuracy": overall_accuracy,
                "unknown_rate": unknown_rate,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "total_samples": total_samples,
                "correct_predictions": correct,
                "unknown_predictions": unknown_predictions_count,
                "known_intent_samples": len(known_intent_data),
                "unknown_intent_samples": len(unknown_intent_texts)
            },
            "per_label": per_label_metrics,
            "unknown_analysis": {
                "true_positives": unknown_tp,  # Unknown correctly identified
                "false_positives": unknown_fp,  # Known incorrectly identified as Unknown
                "true_negatives": unknown_tn,   # Known correctly NOT identified as Unknown
                "false_negatives": unknown_fn,  # Unknown incorrectly identified as known
                "precision": per_label_metrics["Unknown"]["precision"],
                "recall": per_label_metrics["Unknown"]["recall"],
                "f1_score": per_label_metrics["Unknown"]["f1_score"],
                "unknown_detection_rate": unknown_tp / len(unknown_intent_texts) if len(unknown_intent_texts) > 0 else 0.0,
                "false_positive_rate": unknown_fp / len(known_intent_data) if len(known_intent_data) > 0 else 0.0
            },
            "adb_info": {
                "radii": self.intent_radii,
                "labels": self.id2label
            }
        }