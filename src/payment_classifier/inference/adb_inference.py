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
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # CLS token
            probs = outputs.logits.softmax(dim=-1)

        predictions = []
        for i, emb in enumerate(cls_embeddings):
            detected_intents = []
            min_distance = float('inf')
            closest_intent = None
            for label_id, center in self.intent_centers.items():
                distance = torch.norm(emb - center.to(self.device)).item()
                print(f"Label {self.id2label[label_id]}: Distance {distance}, Radius {self.intent_radii[label_id]}")

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
        """Evaluate model performance using ADB on a dataset with ground truth labels."""
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

        # Calculate metrics
        correct = 0
        unknown_predictions = 0

        # For precision/recall/F1 calculation
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        label_stats = defaultdict(lambda: {"correct": 0, "total": 0, "unknown": 0})

        for pred, true_label in zip(all_predictions, all_true_labels):
            true_label_str = self.id2label[true_label]
            pred_label = pred["label"]

            label_stats[true_label_str]["total"] += 1

            if pred_label == "Unknown":
                unknown_predictions += 1
                label_stats[true_label_str]["unknown"] += 1
                # Unknown predictions count as false negatives for the true label
                false_negatives[true_label_str] += 1
            elif pred_label == true_label_str:
                correct += 1
                label_stats[true_label_str]["correct"] += 1
                true_positives[pred_label] += 1
            else:
                # Wrong prediction: false positive for predicted label, false negative for true label
                false_positives[pred_label] += 1
                false_negatives[true_label_str] += 1

        # Calculate overall metrics
        total_samples = len(all_predictions)
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        unknown_rate = unknown_predictions / total_samples if total_samples > 0 else 0.0
        coverage = (total_samples - unknown_predictions) / total_samples if total_samples > 0 else 0.0

        # Calculate per-label metrics including precision, recall, F1
        per_label_metrics = {}
        all_labels = set(self.id2label.values())

        macro_precision_sum = 0
        macro_recall_sum = 0
        macro_f1_sum = 0
        valid_labels = 0

        for label in all_labels:
            if label_stats[label]["total"] > 0:
                tp = true_positives[label]
                fp = false_positives[label]
                fn = false_negatives[label]

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                per_label_metrics[label] = {
                    "accuracy": label_stats[label]["correct"] / label_stats[label]["total"],
                    "coverage": (label_stats[label]["total"] - label_stats[label]["unknown"]) / label_stats[label]["total"],
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "samples": label_stats[label]["total"],
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn
                }

                macro_precision_sum += precision
                macro_recall_sum += recall
                macro_f1_sum += f1
                valid_labels += 1

        # Calculate macro averages
        macro_precision = macro_precision_sum / valid_labels if valid_labels > 0 else 0.0
        macro_recall = macro_recall_sum / valid_labels if valid_labels > 0 else 0.0
        macro_f1 = macro_f1_sum / valid_labels if valid_labels > 0 else 0.0

        return {
            "overall": {
                "accuracy": accuracy,
                "coverage": coverage,
                "unknown_rate": unknown_rate,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "total_samples": total_samples,
                "correct_predictions": correct,
                "unknown_predictions": unknown_predictions
            },
            "per_label": per_label_metrics,
            "adb_info": {
                "radii": self.intent_radii,
                "labels": self.id2label
            }
        }