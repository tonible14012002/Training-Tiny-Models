# 🚀 Iterative LLM Fine-tuning System

A FastAPI-based system for iterative fine-tuning of small language models using synthetic data generation and LLM-based evaluation. Currently implemented for **payment intent classification** as a proof-of-concept, designed for future generalization to arbitrary classification tasks.

---

## 🎯 Current Implementation

### Task Definition (Current: Payment Classification)
- **Payment Intent Classification** with 3 classes:
  - `payment_intent`: User expressing desire to make a payment
  - `payment_request`: User requesting payment from someone else
  - `smart_payment_system_command`: Commands to payment system/bot

> **Future**: Generalizable to any classification task via configurable schemas and prompts

### Architecture Overview
- **FastAPI REST API** for workflow orchestration
- **Teacher LLM**: GPT-4 for synthetic data generation
- **Student Model**: BERT-tiny with LoRA fine-tuning
- **Data Storage**: Local JSONL files with ROUGE-L deduplication

---

## 🔧 Current Components

### 1. Data Generation (`DataGenerator`)
- Uses **PersonaHub dataset** for diverse persona-based generation
- Generates batches of 15 examples per iteration
- **ROUGE-L filtering** (threshold: 0.6) for deduplication
- Iterative generation using previous examples as seeds

### 2. Training Service (`TrainerService`)
- **BERT-tiny** base model with LoRA adapters
- **LoRA config**: rank=16, alpha=32, targets query/value layers
- **Training**: 3 epochs, batch size 8, learning rate 5e-4
- **Checkpointing**: Auto-numbered checkpoints in `.checkpoints/`

### 3. Data Management (`DataManager`)
- Local storage in `.cache/.data.jsonl`
- ROUGE-L based deduplication against existing data
- HuggingFace Dataset conversion for training

### 4. REST API Endpoints
- `POST /workflow/generate-data`: Trigger synthetic data generation
- `POST /workflow/train`: Train student model on current dataset
- `POST /workflow/evaluate`: Model evaluation (placeholder)
- `GET /workflow/status`: Workflow status monitoring
- `GET /workflow/metrics`: Training metrics and history

---

## 🚧 Missing Components (vs. Original Vision)

### Continuous Loop
- **Current**: Manual API triggers
- **Missing**: Autonomous loop with stopping criteria

### Active Learning
- **Current**: Random sampling from PersonaHub
- **Missing**: Error-based sampling, low-confidence targeting

### Evaluation & Monitoring
- **Current**: Basic endpoint stubs
- **Missing**: LLM-generated evaluation sets, automated metric tracking, drift detection

### Human-in-the-Loop
- **Current**: None implemented
- **Missing**: Human validation, review checkpoints

---

## 📁 Project Structure (Current)

### `app/` - FastAPI Application
```
app/
├── main.py                 # FastAPI app with lifespan management
├── api/routes/            # REST API endpoints
│   ├── workflow.py        # Workflow orchestration endpoints
│   └── health.py          # Health check
├── core/
│   ├── services/          # Core business logic
│   │   ├── data_generator/  # Synthetic data generation
│   │   ├── data_manager/    # Data storage & deduplication
│   │   └── trainer/         # Model training with LoRA
│   ├── schemas/           # Pydantic data models
│   └── settings.py        # Configuration management
└── utils/
    └── scorer.py          # ROUGE scoring utilities
```

### `src/` - Shared Payment Classifier Package
```
src/payment_classifier/
├── llm/                   # LLM providers (LiteLLM integration)
├── prompts/               # Prompt management system
└── ...                    # Shared utilities
```

### `training/` - Training Scripts & Data Processing
Legacy training scripts and data processing utilities.

---

## 🔄 Current Workflow

### Setup & Initialization
1. Load human seed data from `.cache/human_seed.json`
2. Initialize GPT-4 teacher LLM and BERT-tiny student
3. Setup FastAPI services with dependency injection

### Data Generation Flow
1. `POST /workflow/generate-data`
2. Load human seed examples
3. Generate synthetic data using PersonaHub personas
4. Apply ROUGE-L deduplication
5. Save to `.cache/.data.jsonl`

### Training Flow
1. `POST /workflow/train`
2. Load accumulated dataset from DataManager
3. Tokenize and preprocess for BERT-tiny
4. Fine-tune with LoRA adapters
5. Save checkpoint to `.checkpoints/{n}/`

---

## 🎛️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: For GPT-4 teacher LLM
- `LOG_LEVEL`: Logging verbosity
- `ENVIRONMENT`: dev/prod environment

### Key Parameters
- **Generation**: 15 examples per iteration, max 20 personas
- **Deduplication**: ROUGE-L threshold 0.6
- **Training**: 3 epochs, 5e-4 learning rate, LoRA rank 16
- **LoRA**: Targets BERT query/value layers, 10% dropout

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Poetry for dependency management
- OpenAI API key for GPT-4 access

### Installation
```bash
# Install dependencies
poetry install --with train

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export LOG_LEVEL="INFO"
```

### Usage
```bash
# Start the API server
python app/main.py

# Generate synthetic data
curl -X POST http://localhost:8000/workflow/generate-data

# Train the model
curl -X POST http://localhost:8000/workflow/train

# Check status
curl http://localhost:8000/workflow/status
```

---

## 🔮 Future Roadmap

### Phase 1: Task Generalization
- Configurable task schemas and label sets
- Dynamic prompt template system
- Generic classification framework

### Phase 2: LLM-Based Evaluation
- Teacher LLM-generated test sets
- Automated evaluation pipeline
- Multi-metric performance tracking

### Phase 3: Enhanced Active Learning
- Low-confidence sample identification
- Error taxonomy-based generation
- Curriculum learning implementation

### Phase 4: Autonomous Loop
- Continuous training pipeline
- LLM-based stopping criteria
- Performance-based iteration control

### Phase 5: Advanced Features
- Multi-task learning capabilities
- Distributed training support
- Production monitoring and alerting