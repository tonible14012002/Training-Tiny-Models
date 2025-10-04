# 🚀 Iterative LLM Fine-tuning System

A FastAPI-based system for iterative fine-tuning of small language models using synthetic data generation and LLM-based evaluation. Currently implemented for **payment intent classification** as a proof-of-concept, designed for future generalization to arbitrary classification tasks.

---

## 🎯 Current Implementation

### Task Definition (Current: Payment Classification)
- **Payment Intent Classification** with 4 classes:
  - `PAYMENT_REQUEST`: User asking someone to send them money
  - `PAYMENT_SEND`: User intends to send/pay money to someone
  - `PAYMENT_COMMAND`: User instructing a system to make a payment
  - `NO_PAYMENT`: No payment intention present

> **Future**: Generalizable to any classification task via configurable schemas and prompts

### Architecture Overview
- **FastAPI REST API** for workflow orchestration
- **Teacher LLM**: GPT-4 for synthetic data generation and evaluation
- **Student Model**: BERT-tiny with LoRA fine-tuning
- **ADB Classification**: Angular Distance-Based classification with intent centers and radii
- **Data Storage**: Local JSONL files with ROUGE-L deduplication
- **Automated Pipeline**: TrainingOrchestrator for continuous improvement loops

---

## 🔧 Current Components

### 1. Data Generation (`DataGenerator` + `EvalGenerator`)
- **Training Data**: Uses PersonaHub dataset for diverse persona-based generation
- **Evaluation Data**: Generates challenging test cases for model evaluation
- Generates batches of 15 examples per iteration
- **ROUGE-L filtering** (threshold: 0.6) for deduplication
- Iterative generation using previous examples as seeds

### 2. Training Service (`TrainerService`)
- **BERT-tiny** base model with LoRA adapters
- **LoRA config**: rank=16, alpha=32, targets query/value layers
- **Training**: 3 epochs, batch size 8, learning rate 5e-4
- **Checkpointing**: Auto-numbered checkpoints with ADB data

### 3. Data Management (`DataManager` + `EvalDataManager`)
- **Training Data**: Local storage in `.cache/.data.jsonl`
- **Evaluation Data**: Versioned storage in `.cache/eval/`
- ROUGE-L based deduplication against existing data
- HuggingFace Dataset conversion for training

### 4. Model Analysis (`ModelAnalyzer`)
- **ADB Evaluation**: Angular Distance-Based classification with intent centers
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, coverage
- **Error Analysis**: Categorizes misclassifications into error buckets
- **Open Intent Detection**: Identifies out-of-distribution samples

### 5. Training Orchestrator (`TrainingOrchestrator`)
- **Automated Pipeline**: Coordinates data generation, training, and evaluation
- **Iterative Loop**: Generate Data → Train → Evaluate → Analyze → Repeat
- **Convergence Detection**: Monitors metrics and stops when improvement plateaus
- **Configuration Management**: Flexible pipeline configuration

### 6. REST API Endpoints
- `POST /workflow/generate-data`: Trigger synthetic data generation
- `POST /workflow/train`: Train student model on current dataset
- `POST /workflow/evaluate`: Model evaluation with comprehensive analysis
- `POST /workflow/start-auto-pipeline`: Start automated training pipeline
- `GET /workflow/status`: Workflow status monitoring
- `GET /workflow/metrics`: Training metrics and history

---

## ✅ Recent Improvements & 🚧 Remaining Gaps

### ✅ Recently Added
- **Automated Pipeline**: TrainingOrchestrator provides autonomous training loops
- **Comprehensive Evaluation**: ModelAnalyzer with ADB classification and error analysis
- **Evaluation Data Generation**: EvalGenerator creates challenging test cases
- **Error Categorization**: Systematic bucketing of misclassification patterns
- **Open Intent Detection**: Identifies out-of-distribution samples

### 🚧 Still Missing (vs. Original Vision)

#### Active Learning Enhancement
- **Current**: Basic error analysis and categorization
- **Missing**: Sophisticated confidence-based sampling, curriculum learning

#### Advanced Error Reasoning
- **Current**: Rule-based error bucketing
- **Missing**: Deep semantic analysis of failure patterns, LLM-powered error explanation

#### Human-in-the-Loop
- **Current**: None implemented
- **Missing**: Human validation workflows, expert review interfaces

#### Real-time Monitoring
- **Current**: Basic pipeline metrics
- **Missing**: Drift detection, performance alerting, production monitoring

---

## 📁 Project Structure (Current)

### `app/` - FastAPI Application
```
app/
├── main.py                    # FastAPI app with lifespan management
├── api/routes/               # REST API endpoints
│   ├── workflow.py           # Workflow orchestration endpoints
│   └── health.py             # Health check
├── core/
│   ├── services/             # Core business logic
│   │   ├── data_generator/   # Training & evaluation data generation
│   │   ├── data_manager/     # Training data storage & deduplication
│   │   ├── eval_data_manager/ # Evaluation data management
│   │   ├── trainer/          # Model training with LoRA + ADB
│   │   ├── model_analyzer/   # ADB evaluation & error analysis
│   │   └── orchestrator/     # Automated training pipeline
│   ├── schemas/              # Pydantic data models
│   ├── mixins/               # Shared utilities and mixins
│   └── settings.py           # Configuration management
└── utils/
    └── scorer.py             # ROUGE scoring utilities
```

### `src/` - Shared Payment Classifier Package
```
src/payment_classifier/
├── llm/                   # LLM providers (LiteLLM integration)
├── prompts/               # Prompt management system
├── inference/             # ADB inference implementation
└── ...                    # Shared utilities
```

### `training/` - Training Scripts & Data Processing
Legacy training scripts and data processing utilities.

---

## 🔄 Current Workflows

### Manual Workflow (Individual Steps)

#### Data Generation Flow
1. `POST /workflow/generate-data`
2. Load human seed examples
3. Generate synthetic data using PersonaHub personas
4. Apply ROUGE-L deduplication
5. Save to `.cache/.data.jsonl`

#### Training Flow
1. `POST /workflow/train`
2. Load accumulated dataset from DataManager
3. Tokenize and preprocess for BERT-tiny
4. Fine-tune with LoRA adapters + calculate ADB centers/radii
5. Save checkpoint with ADB data to `.checkpoints/{n}/`

#### Evaluation Flow
1. `POST /workflow/evaluate`
2. Load model checkpoint with ADB parameters
3. Run comprehensive evaluation with error analysis
4. Generate detailed metrics and error buckets
5. Return evaluation results for analysis

### Automated Pipeline (TrainingOrchestrator)

#### Full Pipeline Flow
1. `POST /workflow/start-auto-pipeline`
2. **Initialization**: Load seed data, configure pipeline parameters
3. **Iteration Loop**:
   - Generate new training data (targeted based on previous errors)
   - Train model with accumulated dataset
   - Evaluate model performance and analyze errors
   - Check convergence criteria (F1 improvement, error rates)
   - Continue or stop based on performance gains
4. **Completion**: Return final model with comprehensive analysis

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
- **ADB**: 90th percentile confidence ratio for intent boundaries
- **Pipeline**: Configurable max iterations, convergence thresholds

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

# Evaluate the model
curl -X POST http://localhost:8000/workflow/evaluate

# Start automated pipeline
curl -X POST http://localhost:8000/workflow/start-auto-pipeline

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