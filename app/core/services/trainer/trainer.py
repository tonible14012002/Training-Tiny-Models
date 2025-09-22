from app.core import schemas
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from datasets import Dataset
from peft import get_peft_model
import numpy as np

import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TrainerService:
    CHECKPOINT_DIR = ".checkpoints"
    def __init__(
            self,
            base_model: str
        ):
        self.base_model = base_model
        self.label2id = schemas.PAYMENT_LABEL.to_dict()
        self.id2label = {v: k for k, v in self.label2id.items()}

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
            save_strategy="steps",
            logging_strategy="steps",
            # eval_steps=100,                    # Very frequent evaluation
            save_steps=100,
            learning_rate=5e-4,
            per_device_train_batch_size=8,     # Smaller batches
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,     # Effective batch size = 32
            num_train_epochs=3,                # More epochs
            warmup_ratio=0.15,                 # More warmup
            weight_decay=0.02,                 # Stronger regularization
            # load_best_model_at_end=True,
            report_to="none",
            # metric_for_best_model="eval_f1_weighted",  # Use F1 as best model metric
            # greater_is_better=True
        )

    def _get_next_checkpoint_number(self) -> int:
        """
        Get the next checkpoint number by checking existing checkpoint folders.

        Returns:
            int: Next checkpoint number (1 if no checkpoints exist)
        """
        checkpoint_path = Path(self.CHECKPOINT_DIR)

        # Create checkpoint directory if it doesn't exist
        checkpoint_path.mkdir(exist_ok=True)

        # Find all numeric subdirectories
        existing_numbers = []
        if checkpoint_path.exists():
            for item in checkpoint_path.iterdir():
                if item.is_dir() and item.name.isdigit():
                    existing_numbers.append(int(item.name))

        # Return next number (max + 1, or 1 if none exist)
        return max(existing_numbers, default=0) + 1

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
        peft_model, _, tokenizer, training_args = self.setup()
        tokenized_train_ds = dataset.map(self._preprocess_input(tokenizer, "msg"), batched=True)

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

        # Save to next available checkpoint number
        checkpoint_num = self._get_next_checkpoint_number()
        checkpoint_path = f"{TrainerService.CHECKPOINT_DIR}/{checkpoint_num}"

        logger.info(f"Training completed. Saving to checkpoint: {checkpoint_path}")
        trainer.save_model(checkpoint_path)

        return checkpoint_num
    
    def _preprocess_input(self, tokenizer: AutoTokenizer, field: str = 'msg'):

        def process(ds: Dataset):
            return tokenizer(
                ds[field],
                padding='max_length',
                max_length=128,
                truncation=True,
            )
        
        return process