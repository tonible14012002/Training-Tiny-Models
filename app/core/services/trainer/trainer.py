from app.core import schemas
from app.core.mixins import NumericalFileAccessMixin
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from datasets import Dataset
from peft import get_peft_model
import torch
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TrainerService(NumericalFileAccessMixin):
    CHECKPOINT_DIR = ".checkpoints"

    @property
    def base_directory(self) -> str:
        return self.CHECKPOINT_DIR

    def __init__(
            self,
            base_model: str
        ):
        self.base_model = base_model
        self.label2id = schemas.PAYMENT_LABEL.to_dict()
        self.id2label = schemas.PAYMENT_LABEL.to_id2label()

        self.lora_config = LoraConfig(
            r=16,                    # Increased rank for better capacity (was 8)
            lora_alpha=32,           # Keep alpha=32 (good alpha/r ratio of 2:1)
            task_type=TaskType.SEQ_CLS,
            lora_dropout=0.1,        # Add dropout for regularization
            bias="none",             # No bias adaptation needed for classification
            target_modules=[         # Target key modules for BERT
                "query",
                "value",
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
            learning_rate=2e-4,
            per_device_train_batch_size=8,     # Smaller batches
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,     # Effective batch size = 32
            num_train_epochs=3,                # More epochs
            warmup_ratio=0.15,                 # More warmup
            weight_decay=0.02,                 # Stronger regularization
            report_to="none",
        )
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

    async def train(self, dataset: Dataset):
        # Save to next available checkpoint number
        checkpoint_num = self._get_next_number()
        checkpoint_path = f"{TrainerService.CHECKPOINT_DIR}/{checkpoint_num}"

        self.training_args.output_dir = f"{checkpoint_path}/trainer_outputs"

        peft_model, _, tokenizer, training_args = self.setup()
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

        # Calculate and save ADB centers and radii using ADBModelInference
        logger.info("Calculating ADB centers and radii...")
        self._calc_and_save_adb(checkpoint_path, dataset)
        logger.info(f"ADB data saved to {checkpoint_path}")

        return checkpoint_num
    
    def load_model(self):
        latest_checkpoint_path = self.get_latest_item_path()
        if latest_checkpoint_path is None:
            raise ValueError("No checkpoints available")

        model = AutoModelForSequenceClassification.from_pretrained(str(latest_checkpoint_path))
        tokenizer = AutoTokenizer.from_pretrained(str(latest_checkpoint_path))
        return model, tokenizer

    def _get_preprocessor(self, tokenizer: AutoTokenizer, field: str = 'msg'):
        def process(ds: Dataset):
            return tokenizer(
                ds[field],
                padding='max_length',
                max_length=128,
                truncation=True,
            )
        
        return process

    def _calc_and_save_adb(self, checkpoint_path: str, dataset: Dataset):
        """Calculate ADB centers and radii using ADBModelInference and save them"""
        from src.payment_classifier.inference.adb_inference import ADBModelInference

        # Load the trained model using ADBModelInference
        adb_inference = ADBModelInference(checkpoint_path)

        # Calculate ADB centers and radii
        intent_centers, intent_radii = adb_inference.calc_adb(dataset)

        # Save ADB data
        adb_data = {
            "intent_centers": intent_centers,
            "intent_radii": intent_radii
        }

        adb_file_path = Path(checkpoint_path) / "adb_data.json"
        with open(adb_file_path, 'w') as f:
            json.dump(adb_data, f, indent=2)


