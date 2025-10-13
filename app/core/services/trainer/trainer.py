from app.core.mixins import NumericalFileAccessHelper
from app.core.models.models import LabelConfig
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from datasets import Dataset
from peft import get_peft_model
import torch
import logging
import json
import os
import re
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class TrainerService:
    def __init__(
            self,
            base_model: str,
            label_config: LabelConfig,
            base_dir: str = '.checkpoints'
        ):
        self.base_model = base_model
        self.label_config = label_config
        self.label2id = label_config.get_label2id()
        self.id2label = label_config.get_id2label()

        # Create label-specific checkpoint directory
        label_name = self._sanitize_name(label_config.name)
        self.CHECKPOINT_DIR = f'{base_dir}/{label_name}'

        # Ensure directory exists
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

        # Initialize file access helper
        self._file_helper = NumericalFileAccessHelper(self.CHECKPOINT_DIR)

        self.lora_config = LoraConfig(
            r=16,                    # Increased rank for better capacity (was 8)
            lora_alpha=32,           # Keep alpha=32 (good alpha/r ratio of 2:1)
            task_type=TaskType.SEQ_CLS,
            lora_dropout=0.1,        # Add dropout for regularization
            bias="none",             # No bias adaptation needed for classification
            target_modules=[         # Target key modules for BERT
                "query",
                "value",
                "dense"
            ],
        )

        self.training_args = TrainingArguments(
            # eval_strategy="steps",
            # eval_steps=100,                    # Very frequent evaluation
            # load_best_model_at_end=True,
            # metric_for_best_model="eval_f1_weighted",  # Use F1 as best model metric
            # greater_is_better=True
            save_strategy="steps",
            logging_strategy="steps",
            output_dir=self.CHECKPOINT_DIR,
            # save_steps=100,
            learning_rate=2e-5,
            # per_device_train_batch_size=8,     # Smaller batches
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,     # Effective batch size = 32
            num_train_epochs=3,                # More epochs
            warmup_ratio=0.15,                 # More warmup
            # weight_decay=0.02,                 # Stronger regularization
            report_to="none",
        )

    def _sanitize_name(self, name: str) -> str:
        """Sanitize label config name for use in file paths"""
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    def setup(self):
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

        model = AutoModelForSequenceClassification.from_pretrained(self.base_model, label2id=self.label2id, id2label=self.id2label).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        peft_model = get_peft_model(model, self.lora_config)

        return [
            peft_model,
            model,
            tokenizer,
            self.training_args
        ]

    async def train(self, dataset: Dataset, inference_type: str = "prob",return_full_path: bool = False) -> str:
        # Save to next available checkpoint number
        checkpoint_num = self._file_helper._get_next_number()
        checkpoint_path = f"{self.CHECKPOINT_DIR}/{checkpoint_num}"

        self.training_args.output_dir = f"{checkpoint_path}/trainer_outputs"

        peft_model, _, tokenizer, training_args = self.setup()
        seed = random.randint(0, 10000)
        dataset = dataset.shuffle(seed=seed)
        tokenized_train_ds = dataset.map(self._get_preprocessor(tokenizer, "msg"), batched=True)

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            tokenizer=tokenizer,
        )
        # Load base model
        logger.debug("Base model loaded")
        # Load Lora Adapters

        trainer.train()
        logger.info(f"Training completed. Saving to checkpoint: {checkpoint_path}")
        trainer.save_model(checkpoint_path)
        
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(checkpoint_path + "/_merged")

        # Perform post-training calculations based on inference type
        if inference_type == "adb":
            logger.info("Calculating ADB centers and radii...")
            self._post_train_adb(checkpoint_path, dataset)
            logger.info(f"ADB data saved to {checkpoint_path}")
        elif inference_type == "prob":
            logger.info("Saving inference type for probability-based inference...")
            self._post_train_prob(checkpoint_path)
            logger.info(f"Probability-based inference config saved to {checkpoint_path}")

        if return_full_path:
            return checkpoint_path
        return checkpoint_num

    async def continual_train(
        self,
        checkpoint_id: str,
        dataset: Dataset,
        inference_type: str = "prob"
    ) -> str:
        """Continue training from an existing checkpoint with sub-versioning.

        Args:
            checkpoint_id: The checkpoint identifier to continue from (e.g., "10", "10.1", "10.2")
            dataset: The dataset to train on
            inference_type: Type of inference ("adb" or "prob")

        Returns:
            The new sub-checkpoint identifier (e.g., "10.1", "10.2")
        """
        # Determine paths using the new method that supports any checkpoint ID
        load_from_path, save_to_path = self._file_helper.get_next_sub_version_from_id(checkpoint_id)

        # Verify source checkpoint exists
        if not Path(load_from_path).exists():
            raise ValueError(f"Checkpoint {load_from_path} does not exist")

        logger.info(f"Continuing training from checkpoint: {load_from_path}")
        logger.info(f"Will save to: {save_to_path}")

        # Set up training args with new output directory
        self.training_args.output_dir = f"{save_to_path}/trainer_outputs"

        # Load base model and existing PEFT adapters
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

        # Load the base model first
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            label2id=self.label2id,
            id2label=self.id2label
        ).to(device)

        # Load the PEFT adapters from the checkpoint
        model = PeftModel.from_pretrained(base_model, load_from_path)

        tokenizer = AutoTokenizer.from_pretrained(load_from_path)

        # Prepare dataset
        seed = random.randint(0, 10000)
        dataset = dataset.shuffle(seed=seed)
        tokenized_train_ds = dataset.map(self._get_preprocessor(tokenizer, "msg"), batched=True)

        # Create trainer with loaded model
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_train_ds,
            tokenizer=tokenizer,
        )

        logger.info("Starting continual training...")
        trainer.train()

        logger.info(f"Training completed. Saving to checkpoint: {save_to_path}")
        trainer.save_model(save_to_path)

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(save_to_path + "/_merged")

        # Perform post-training calculations based on inference type
        if inference_type == "adb":
            logger.info("Calculating ADB centers and radii...")
            self._post_train_adb(save_to_path, dataset)
            logger.info(f"ADB data saved to {save_to_path}")
        elif inference_type == "prob":
            logger.info("Saving inference type for probability-based inference...")
            self._post_train_prob(save_to_path)
            logger.info(f"Probability-based inference config saved to {save_to_path}")

        # Extract and return the checkpoint identifier
        checkpoint_id = Path(save_to_path).name
        return checkpoint_id

    def _get_preprocessor(self, tokenizer: AutoTokenizer, field: str = 'msg'):
        def process(ds: Dataset):
            return tokenizer(
                ds[field],
                padding='max_length',
                max_length=128,
                truncation=True,
            )
        
        return process

    def _post_train_adb(self, checkpoint_path: str, dataset: Dataset):
        """Calculate ADB centers and radii using ADBModelInference and save them"""
        from src.payment_classifier.inference.adb_inference import ADBModelInference

        # Load the trained model using ADBModelInference with label config
        adb_inference = ADBModelInference(checkpoint_path, self.label_config)

        # Calculate ADB centers and radii
        intent_centers, intent_radii = adb_inference.post_train(dataset)

        # Save ADB data
        adb_data = {
            "intent_centers": intent_centers,
            "intent_radii": intent_radii
        }

        adb_file_path = Path(checkpoint_path) / "adb_data.json"
        with open(adb_file_path, 'w') as f:
            json.dump(adb_data, f, indent=2)

        # Save inference type metadata
        inference_config = {
            "inference_type": "adb",
            "created_at": str(Path(checkpoint_path).stat().st_mtime)
        }

        config_file_path = Path(checkpoint_path) / "inference_config.json"
        with open(config_file_path, 'w') as f:
            json.dump(inference_config, f, indent=2)

    def _post_train_prob(self, checkpoint_path: str):
        """Save inference type metadata for probability-based inference"""
        # Save inference type metadata
        inference_config = {
            "inference_type": "prob",
            "created_at": str(Path(checkpoint_path).stat().st_mtime)
        }

        config_file_path = Path(checkpoint_path) / "inference_config.json"
        with open(config_file_path, 'w') as f:
            json.dump(inference_config, f, indent=2)
