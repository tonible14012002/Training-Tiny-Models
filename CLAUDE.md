# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Instructions

- **Work ONLY in `app/api/routes/v2/` folder** (new endpoints). Old endpoints in `app/api/routes/` are for reference only.

## Running the Application

### Start Development Server
```bash
make start
# or
python app/main.py
# or
./scripts/run_server.sh
```

### Run Migrations
```bash
python cmd/run_migrations.py
```

### Run Specific Command Scripts
```bash
make run-cmd file=<filename_without_extension>
# Example: make run-cmd file=test_models
```

## Architecture Overview

This is an **iterative LLM fine-tuning system** for text classification using:
- **Teacher LLM**: GPT-4 for synthetic data generation
- **Student Model**: BERT-tiny with LoRA fine-tuning
- **Database**: SQLite with SQLModel ORM
- **API**: FastAPI with async/await patterns

### Core Pipeline Stages

The system implements a 4-stage pipeline (see `docs/pipeline_stages.md` for details):

1. **First-Gen** (`phase_number=0`): Generate initial training data using human seed examples
   - Creates Phase → ComposalDataset → DatasetFiles
   - Uses LLM with persona-based generation
   - Applies ROUGE-L deduplication (threshold 0.6)
   - Saves incrementally with batch tracking

2. **Train**: Fine-tune BERT-tiny model with LoRA adapters
   - Loads dataset from phase's composal dataset
   - 3 epochs, learning rate 2e-5, LoRA rank 16
   - Saves merged model to `.checkpoints/{pipeline_id}/{phase_number}/`

3. **Evaluate**: Test model on frozen test set and training pool
   - Identifies error samples and low-confidence predictions
   - Generates comprehensive metrics (accuracy, precision, recall, F1)
   - Stores evaluation results with per-label metrics

4. **Generate Fix** (Partial): Analyze errors and prepare targeted data generation
   - Categorizes errors into error buckets using LLM
   - Prepares generation config (NOT YET IMPLEMENTED: actual fix data generation)

### Data Flow

```
Create Pipeline → Create Phase → Generate Data (First-Gen) → Train Model →
Evaluate Model → Analyze Errors → Generate Fix Data → (loop back to Train)
```

## Key Components

### Database Layer (`app/core/models/models.py`)

Core entities (see `DATABASE_SCHEMA.md` for full schema):
- **Pipeline**: Root entity with label configuration
- **PipelinePhase**: Tracks training phases with hierarchical `phase_path` (e.g., "" for root, "parent_id" for children)
- **ComposalDataset**: Collection of dataset files for a phase
- **DatasetFile**: Individual dataset files (train/validation/test/generated)
- **ErrorBucket**: Categorizes model error types
- **TrainedModel**: Stores trained model metadata and checkpoints
- **EvaluationResult**: Stores evaluation metrics and error samples

**Important**: Many fields store JSON as strings (e.g., `label_config.id2label`, `training_params`) due to SQLite limitations.

### Repository Layer (`app/core/repositories/`)

Repository pattern for database operations:
- `pipeline_repository.py`: Pipeline CRUD
- `phase_repository.py`: Phase management and querying
- `dataset_repository.py`: Dataset and file operations
- `error_bucket_repository.py`: Error bucket management
- `evaluation_result_repository.py`: Evaluation storage
- `trained_model_repository.py`: Model checkpoint tracking

All repositories use **async SQLAlchemy sessions**.

### Service Layer (`app/core/services/`)

Business logic services:
- `data_generator/`: Synthetic data generation with LLM (`DataGeneratorV2`)
- `data_manager/`: Dataset storage and ROUGE-L deduplication (`DataManager`)
- `trainer/`: LoRA model training (`TrainerService`)
- `error_categorizer/`: LLM-based error analysis
- `orchestrator/`: Automated pipeline coordination
- `prompt_builder/`: Dynamic prompt construction

### API Layer (`app/api/routes/v2/`)

**V2 endpoints** (actively developed):
- `POST /workflow/pipeline`: Create new pipeline with label config
- `POST /workflow/first-gen`: Generate initial training data
- `POST /workflow/train`: Train model on phase dataset
- `POST /workflow/evaluate`: Evaluate trained model
- `POST /workflow/classify-errors`: Categorize evaluation errors into buckets

Helper functions:
- `load_human_seed(label_config)`: Load human seed samples from `.cache/human_seed.json`
- `load_frozen_set(label_config)`: Load frozen test set from `.cache/frozen_test_set.json`

## Database Migrations

Migration files in `app/migrations/` follow pattern: `{version}_{description}.sql`

- **Create migration**: Add new `.sql` file with next version number
- **Run migrations**: `python cmd/run_migrations.py`
- **Check applied**: `sqlite3 pipeline.db "SELECT * FROM schema_migrations;"`
- **Reset database**: `rm pipeline.db && python cmd/run_migrations.py`

Migrations use SQLite syntax:
- `UUID` → `TEXT`
- `JSONB` → `TEXT`
- Use `IF NOT EXISTS` for idempotency
- Use `ON CONFLICT(column)` (no space before parenthesis)

## Key Patterns

### Phase Hierarchy via `phase_path`

Phases organize into sequences using `phase_path`:
- `phase_path=""` → Root phase (first in sequence)
- `phase_path="parent_id"` → First generation child
- `phase_path="p1_id/p2_id"` → Second generation child

All phases with same root phase ID (first segment of path) belong to the same sequence.

### Async Repository Pattern

```python
from app.api.dependencies import get_db_session
from app.core import repositories as repos

async def my_endpoint(db: AsyncSession = Depends(get_db_session)):
    pipeline_repo = repos.PipelineRepository(db)
    pipeline = await pipeline_repo.get_by_id(pipeline_id)
    # ... work with pipeline
    await db.commit()
```

### Label Configuration

```python
# label_config stores JSON as strings
id2label = label_config.get_id2label()  # Returns dict: {0: "label_name", ...}
label2id = label_config.get_label2id()  # Returns dict: {"label_name": 0, ...}
```

### Data Generation with Callbacks

```python
async def on_batch_generated(batch_num: int, samples: List[Sample], temp_file: str):
    # Called after each batch generation
    # Save batch file record to database
    pass

samples, base_file = await data_generator.fresh_gen_v2(
    human_seeds=seeds,
    expect_total_each_label={"label1": 200, "label2": 200},
    on_batch_generated=on_batch_generated
)
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for GPT-4 teacher LLM
- `LOG_LEVEL`: Logging verbosity (default: INFO)
- `PYTHONPATH`: Must be set to `./` (handled by Makefile)

### Key Directories
- `.cache/`: Human seeds, frozen test set, temporary data
- `.checkpoints/`: Trained model checkpoints (organized by pipeline_id/phase_number)
- `app/migrations/`: Database migration SQL files
- `pipeline.db`: SQLite database file (git-ignored)

## Important Notes

### V1 vs V2 Architecture

- **V1** (legacy): Located in `app/api/routes/workflow.py` - reference only
- **V2** (current): Located in `app/api/routes/v2/workflow.py` - actively developed
  - Uses database-backed pipeline/phase tracking
  - Supports hierarchical phase management via `phase_path`
  - Implements incremental batch generation with callbacks
  - Tracks dataset files separately from composal datasets

### Phase Status Lifecycle

Phases progress through states: `pending` → `in_progress` → `completed` (or `failed`)

### Dataset File Status

Dataset files track generation progress: `generating` → `done`

### Error Buckets

Error buckets categorize model failures for targeted data generation. Currently used in evaluation but not yet integrated into fix data generation.

### Testing

- Test frozen set stored in `.cache/frozen_test_set.json`
- Human seed samples in `.cache/human_seed.json`
- Use `cmd/test_models.py` for model testing

## Common Workflows

### Create and Run New Pipeline

1. Create pipeline with label config: `POST /workflow/pipeline`
2. Generate initial data: `POST /workflow/first-gen`
3. Train model: `POST /workflow/train`
4. Evaluate model: `POST /workflow/evaluate`
5. Classify errors: `POST /workflow/classify-errors`
6. (Future) Generate fix data and repeat from step 3

### Database Schema Changes

1. Create migration file: `app/migrations/00X_description.sql`
2. Write SQL (use SQLite syntax, `IF NOT EXISTS`)
3. Run migrations: `python cmd/run_migrations.py`
4. Update SQLModel classes in `app/core/models/models.py`
5. Update repositories if needed

### Add New V2 Endpoint

1. Define request/response schemas in `app/core/schemas/workflow.py`
2. Add repository methods if needed in `app/core/repositories/`
3. Implement endpoint in `app/api/routes/v2/workflow.py`
4. Use async patterns with `Depends(get_db_session)`
5. Handle errors and return proper status codes
