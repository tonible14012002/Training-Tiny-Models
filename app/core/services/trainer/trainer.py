from app.core import schemas
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model
import torch

class TrainerService:
    def __init__(
            self,
            base_model: str
        ):
        self.base_model = base_model
        self.label2id = {
            schemas.PAYMENT_LABEL.PAYMENT_INTENT: 1,
            schemas.PAYMENT_LABEL.PAYMENT_REQUEST: 2,
            schemas.PAYMENT_LABEL.PAYMENT_COMMAND: 3,
        }
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
                "key",
                "dense"
            ],
        )
    
    def setup(self):
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

        model = AutoModelForSequenceClassification.from_pretrained(self.base_model, label2id=self.label2id, id2label=self.id2label).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        peft_model = get_peft_model(model, self.lora_config)
        peft_model.print_trainable_parameters()

    async def train(self):
        # Load base model
        # Load Lora Adapters
        pass