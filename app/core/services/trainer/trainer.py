from app.core.mixins import NumericalFileAccessHelper
from app.core.models.models import LabelConfig
from app.core.schemas.workflow import TrainingConfig
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
from typing import Optional, Dict, Any, Union

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

        # Default training configuration
        self.default_training_config = {
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 3,
            "warmup_ratio": 0.15,
            "weight_decay": 0.01,
            "save_strategy": "steps",
            "logging_strategy": "steps",
            "eval_strategy": "no",
            "save_steps": 500,
            "logging_steps": 10,
            "report_to": "none",
            "seed": 42,
        }

    def _sanitize_name(self, name: str) -> str:
        """Sanitize label config name for use in file paths"""
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    def _merge_training_config(
        self,
        custom_config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Merge default training config with custom config.

        Args:
            custom_config: Custom training configuration (TrainingConfig or dict)

        Returns:
            Merged configuration dictionary
        """
        # Start with default config
        merged_config = self.default_training_config.copy()

        if custom_config is None:
            return merged_config

        # Convert TrainingConfig to dict if needed
        if isinstance(custom_config, TrainingConfig):
            custom_dict = custom_config.model_dump(exclude_none=True)
        else:
            custom_dict = custom_config.copy()

        # Handle parameter name compatibility (evaluation_strategy -> eval_strategy)
        if "evaluation_strategy" in custom_dict:
            custom_dict["eval_strategy"] = custom_dict.pop("evaluation_strategy")

        # Merge custom config, overwriting defaults
        merged_config.update(custom_dict)

        return merged_config

    def _build_training_arguments(
        self,
        output_dir: str,
        config: Dict[str, Any]
    ) -> TrainingArguments:
        """
        Build TrainingArguments from configuration dictionary.

        Args:
            output_dir: Directory to save training outputs
            config: Training configuration dictionary

        Returns:
            TrainingArguments instance
        """
        # Add output_dir to config
        config_with_output = config.copy()
        config_with_output["output_dir"] = output_dir

        return TrainingArguments(**config_with_output)

    def export_training_config_as_json(
        self,
        config: Dict[str, Any],
        include_lora: bool = True
    ) -> str:
        """
        Export training configuration as JSON string for database storage.

        Args:
            config: Training configuration dictionary
            include_lora: Whether to include LoRA configuration

        Returns:
            JSON string of the configuration
        """
        # Helper function to make objects JSON serializable
        def make_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj

        export_config = {
            "training_args": make_serializable(config.copy()),
        }

        if include_lora:
            export_config["lora_config"] = {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "task_type": str(self.lora_config.task_type),
                "lora_dropout": self.lora_config.lora_dropout,
                "bias": self.lora_config.bias,
                "target_modules": list(self.lora_config.target_modules) if isinstance(self.lora_config.target_modules, set) else self.lora_config.target_modules,
            }

        return json.dumps(export_config, indent=2)

    def setup(self):
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

        model = AutoModelForSequenceClassification.from_pretrained(self.base_model, label2id=self.label2id, id2label=self.id2label).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        peft_model = get_peft_model(model, self.lora_config)

        return peft_model, model, tokenizer

    async def train(
        self,
        dataset: Dataset,
        inference_type: str = "prob",
        return_full_path: bool = False,
        custom_training_config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None
    ) -> tuple[str, str]:
        """
        Train a model on the given dataset.

        Args:
            dataset: Dataset to train on
            inference_type: Type of inference ("adb" or "prob")
            return_full_path: Whether to return full path or just checkpoint number
            custom_training_config: Custom training configuration (TrainingConfig or dict)

        Returns:
            Tuple of (checkpoint_path_or_num, training_config_json)
        """
        # Save to next available checkpoint number
        checkpoint_num = self._file_helper._get_next_number()
        checkpoint_path = f"{self.CHECKPOINT_DIR}/{checkpoint_num}"
        output_dir = f"{checkpoint_path}/trainer_outputs"

        # Merge configs
        merged_config = self._merge_training_config(custom_training_config)

        # Build TrainingArguments
        training_args = self._build_training_arguments(output_dir, merged_config)

        peft_model, _, tokenizer = self.setup()

        # Use seed from config or generate random
        seed = merged_config.get("seed", random.randint(0, 10000))
        dataset = dataset.shuffle(seed=seed)
        tokenized_train_ds = dataset.map(self._get_preprocessor(tokenizer, "msg"), batched=True)

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            tokenizer=tokenizer,
        )

        logger.info(f"Starting training with config: {merged_config}")
        trainer.train()
        logger.info(f"Training completed. Saving to checkpoint: {checkpoint_path}")
        trainer.save_model(checkpoint_path)

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(checkpoint_path + "/_merged")

        # Export and save training config
        config_json = self.export_training_config_as_json(merged_config, include_lora=True)
        config_file_path = Path(checkpoint_path) / "training_config.json"
        with open(config_file_path, 'w') as f:
            f.write(config_json)
        logger.info(f"Training config saved to {config_file_path}")

        # Perform post-training calculations based on inference type
        if inference_type == "adb":
            logger.info("Calculating ADB centers and radii...")
            self._post_train_adb(checkpoint_path, dataset)
            logger.info(f"ADB data saved to {checkpoint_path}")
        elif inference_type == "prob":
            logger.info("Saving inference type for probability-based inference...")
            self._post_train_prob(checkpoint_path)
            logger.info(f"Probability-based inference config saved to {checkpoint_path}")

        result_path = checkpoint_path if return_full_path else checkpoint_num
        return result_path, config_json

    async def continual_train(
        self,
        checkpoint_id: str,
        dataset: Dataset,
        return_full_path: bool = False,
        inference_type: str = "prob",
        custom_training_config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None
    ) -> tuple[str, str]:
        """Continue training from an existing checkpoint with sub-versioning.

        Args:
            checkpoint_id: The checkpoint identifier to continue from (e.g., "10", "10.1", "10.2")
                          OR a full path to a checkpoint directory
            dataset: The dataset to train on
            return_full_path: Whether to return full path or just checkpoint identifier
            inference_type: Type of inference ("adb" or "prob")
            custom_training_config: Custom training configuration (TrainingConfig or dict)

        Returns:
            Tuple of (checkpoint_path_or_id, training_config_json)
        """
        # Check if checkpoint_id is a full path or a checkpoint identifier
        if os.path.isabs(checkpoint_id) or os.path.sep in checkpoint_id:
            # It's a full path - use it directly as load path
            load_from_path = checkpoint_id
            # Generate new checkpoint in standard directory
            checkpoint_num = self._file_helper._get_next_number()
            save_to_path = f"{self.CHECKPOINT_DIR}/{checkpoint_num}"
        else:
            # It's a checkpoint ID - use existing logic
            load_from_path, save_to_path = self._file_helper.get_next_sub_version_from_id(checkpoint_id)

        # Verify source checkpoint exists
        if not Path(load_from_path).exists():
            raise ValueError(f"Checkpoint {load_from_path} does not exist")

        logger.info(f"Continuing training from checkpoint: {load_from_path}")
        logger.info(f"Will save to: {save_to_path}")

        # Merge configs
        merged_config = self._merge_training_config(custom_training_config)
        output_dir = f"{save_to_path}/trainer_outputs"

        # Build TrainingArguments
        training_args = self._build_training_arguments(output_dir, merged_config)

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
        seed = merged_config.get("seed", random.randint(0, 10000))
        dataset = dataset.shuffle(seed=seed)
        tokenized_train_ds = dataset.map(self._get_preprocessor(tokenizer, "msg"), batched=True)

        # Create trainer with loaded model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            tokenizer=tokenizer,
        )

        logger.info(f"Starting continual training with config: {merged_config}")
        trainer.train()

        logger.info(f"Training completed. Saving to checkpoint: {save_to_path}")
        trainer.save_model(save_to_path)

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(save_to_path + "/_merged")

        # Export and save training config
        config_json = self.export_training_config_as_json(merged_config, include_lora=True)
        config_file_path = Path(save_to_path) / "training_config.json"
        with open(config_file_path, 'w') as f:
            f.write(config_json)
        logger.info(f"Training config saved to {config_file_path}")

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
        checkpoint_id_result = Path(save_to_path).name

        result_path = save_to_path if return_full_path else checkpoint_id_result
        return result_path, config_json

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
