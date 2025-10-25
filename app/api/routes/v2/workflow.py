from fastapi import APIRouter, Request, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.core import repositories as repos
from app.api.dependencies import get_db_session
from app.core import services
from app.core import schemas
from src.payment_classifier.inference.prob_inference import ProbModelInference
from app.utils.dataset_helper import DatasetHelper
from datasets import concatenate_datasets
from pathlib import Path
import shutil
import random

import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflow")

def load_human_seed(label_config):
    with open("./.cache/human_seed.json", "r") as f:
        human_seed = json.loads(f.read())

    human_samples = [
        schemas.Sample(
            msg=msg["msg"],
            label=label_config.get_id2label()[str(msg["label"])],
        ) for msg in human_seed
    ]

    return human_samples


def load_frozen_set(label_config, cache_path=".cache"):
    frozen_set_path = f"{cache_path}/frozen_test_set.json"
    with open(frozen_set_path, "r") as f:
        frozen_set = json.loads(f.read())

    eval_ds = DatasetHelper.json_to_ds(
        json_list=frozen_set,
        label_config=label_config
    )
    return eval_ds


@router.post("/pipeline")
async def create_pipeline(
    payload: schemas.CreatePipelineRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new pipeline with label configuration"""
    pipeline_repo = repos.PipelineRepository(db)

    # Create pipeline
    pipeline = await pipeline_repo.create(name=payload.name)

    # Create label config
    label_config = await pipeline_repo.create_label_config(
        pipeline_id=pipeline.id,
        name=payload.label_config.name,
        id2label=json.dumps(payload.label_config.id2label),
        label2id=json.dumps(payload.label_config.label2id),
        label_explanation=json.dumps(payload.label_config.label_explanation) if payload.label_config.label_explanation else None
    )

    return {
        "message": "Pipeline created successfully",
        "data": {
            "id": pipeline.id,
            "name": pipeline.name,
            "created_at": pipeline.created_at.isoformat(),
            "updated_at": pipeline.updated_at.isoformat(),
            "label_config": {
                "id": label_config.id,
                "name": label_config.name,
                "id2label": label_config.get_id2label(),
                "label2id": label_config.get_label2id(),
                "label_explanation": label_config.get_label_explanation(),
                "created_at": label_config.created_at.isoformat(),
            }
        }
    }

@router.get("/pipelines")
async def list_pipelines(
    db: AsyncSession = Depends(get_db_session),
):
    """List all pipelines"""
    pipeline_repo = repos.PipelineRepository(db)
    pipelines = await pipeline_repo.get_all()

    return {
        "message": "Pipelines retrieved successfully",
        "data": [
            {
                "id": p.id,
                "name": p.name,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
                "label_config": {
                    "id": p.label_configs[0].id,
                    "name": p.label_configs[0].name,
                    "id2label": p.label_configs[0].get_id2label(),
                    "label2id": p.label_configs[0].get_label2id(),
                    "label_explanation": p.label_configs[0].get_label_explanation(),
                    "created_at": p.label_configs[0].created_at.isoformat(),
                } if p.label_configs else None
            }
            for p in pipelines
        ]
    }

@router.post("/convert/onnx")
async def convert_to_onnx(
    pipeline_id: str = Body(..., description="The ID of the pipeline to convert"),
    path: str = Body(..., description="The path to save the ONNX model"),
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    return {
        "message": "Not implemented yet",
    }

@router.post("/phase/first-gen")
async def first_generation(
    payload: schemas.StartPipelineRequest,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    '''
    1. Create 1st phase in the pipeline
    2. Create composal dataset and dataset file records FIRST (before generation)
    3. Generate initial dataset using DataGeneratorV2 with batch tracking
    4. Update final dataset file path and info after generation
    5. Return phase info, dataset file info, and batch files
    '''
    # Fetch pipeline and label config from database
    pipeline = await repos.PipelineRepository(db).get_by_id(pipeline_id=payload.pipeline_id)

    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}

    # Create new iteration phase (base phase with phase_path="")
    phase = await repos.PhaseRepository(db).create(
        pipeline_id=pipeline.id,
        status=schemas.PHASE_STATUS.IN_PROGRESS,
        phase_number=0,
        previous_phase_id=None,  # No previous phase - this is a base phase
    )

    label_config = pipeline.label_configs[0]  # Get the first label config

    # Create composal directory
    composal_dir = Path(f'.cache/{pipeline.id}/{phase.id}/composal')
    composal_dir.mkdir(parents=True, exist_ok=True)
    composal_file_path = composal_dir / f'composal_ds.jsonl'

    # Create Dataset and DatasetFile records FIRST (before generation)
    base_ds = await repos.DatasetRepository(db).create_composal_ds(
        pipeline_id=pipeline.id,
        phase_id=phase.id,
        name=f"{pipeline.name} Composal Dataset",
        description="Composal Dataset for pipeline",
        file_path=str(composal_file_path),
    )

    ds_file = await repos.DatasetFileRepository(db).create_dataset_file(
        parent_dataset_id=base_ds.id,
        file_path="",  # Will be updated after generation
        file_type="jsonl",
        phase_id=phase.id,
        sample_count=0  # Will be updated after generation
    )

    # Create DataManager and DataGeneratorV2 instances for this pipeline
    data_manager = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.65,
        base_dir=f'.cache/{pipeline.id}/{phase.id}/{phase.phase_number}'
    )

    # Create validator for label correction
    validator = services.DataValidator(
        llm=request.app.state.validator_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        label_config=label_config
    )

    data_generator_v2 = services.DataGeneratorV2(
        llm=request.app.state.teacher_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        data_manager=data_manager,
        high_llm=request.app.state.teacher_llm_high,
        validator=validator
    )

    human_seeds = load_human_seed(label_config)

    # Create dict
    label_count_dict = label_config.get_label2id()
    for key in label_count_dict:
        label_count_dict[key] = 200

    # Define callback to save each batch to database
    async def on_batch_generated(batch_number: int, samples: list, batch_file_path: str, metadata_path: str):
        """Callback to save batch information to database"""
        batch_repo = repos.BatchGeneratedDatasetFileRepository(db)
        await batch_repo.create_batch_file(
            parent_dataset_file_id=ds_file.id,
            file_path=batch_file_path,
            batch_number=batch_number,
            sample_count=len(samples)
        )
        logger.info(f"Saved batch {batch_number} with {len(samples)} samples to database (metadata: {metadata_path})")

    # Generate with batch tracking
    generated, path = await data_generator_v2.fresh_gen_v2(
        human_seeds=human_seeds,
        expect_total_each_label=label_count_dict,
        label_config=label_config,
        on_batch_generated=on_batch_generated
    )

    # Copy final file to composal directory
    source_path = Path(path)
    shutil.copy2(source_path, composal_file_path)

    # Update DatasetFile with final path, sample count, and status
    await repos.DatasetFileRepository(db).update(
        file_id=ds_file.id,
        file_path=path,
        sample_count=len(generated),
        status=schemas.DATASET_FILE_STATUS.DONE
    )

    # Refresh to get updated data
    await db.refresh(ds_file)

    # Get all batch files for this dataset file
    batch_files = await repos.BatchGeneratedDatasetFileRepository(db).get_by_parent_file(ds_file.id)

    return {
        "message": f"Generated {len(generated)} samples in {len(batch_files)} batches",
        "data": {
            "compose_ds": base_ds.model_dump(),
            "ds_file": ds_file.model_dump(),
            "batch_files": [bf.model_dump() for bf in batch_files],
        }
    }

@router.post("/test/first-gen")
async def test_first_generation(
    payload: schemas.TestFirstGenRequest,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    """Test endpoint for first generation without database mutations"""
    # Fetch pipeline and label config from database (read-only)
    pipeline = await repos.PipelineRepository(db).get_by_id(pipeline_id=payload.pipeline_id)

    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}

    label_config = pipeline.label_configs[0]  # Get the first label config

    # Create DataManager and DataGeneratorV2 instances for testing
    data_manager = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.65,
        base_dir=f'{payload.cache_path}/test/first-gen/{payload.pipeline_id}'
    )

    # Create validator for label correction
    validator = services.DataValidator(
        llm=request.app.state.validator_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        label_config=label_config
    )

    data_generator_v2 = services.DataGeneratorV2(
        llm=request.app.state.teacher_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        data_manager=data_manager,
        high_llm=request.app.state.teacher_llm_high,
        validator=validator
    )

    human_seeds = load_human_seed(label_config)

    # Create dict for label counts
    label_count_dict = label_config.get_label2id()
    for key in label_count_dict:
        label_count_dict[key] = 0
    label_count_dict["open_intent"] = 3000

    # Generate data
    data_generator_v2.SEED_PROMPT_KEY = "v2/train/seed_open_intent"
    generated, path = await data_generator_v2.fresh_gen_v2(
        human_seeds=human_seeds,
        expect_total_each_label=label_count_dict,
        label_config=label_config
    )

    return {
        "message": f"Test generation completed: {len(generated)} samples",
        "data": {
            "sample_count": len(generated),
            "file_path": path,
            "cache_path": payload.cache_path,
            "label_counts": {
                label: sum(1 for s in generated if s.label == label)
                for label in label_config.get_id2label().values()
            }
        }
    }

@router.get("/phase/{phase_id}/generation-status")
async def get_generation_status(
    phase_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get dataset generation status for a phase (can be called while generation is in progress)

    Returns:
        - DatasetFile information with hardcoded status "generating"
        - All BatchGeneratedDatasetFile records (even if generation is incomplete)
        - Current sample counts and label distribution from batch files
    """
    # Fetch phase
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {
            "error": "Phase not found",
            "message": f"No phase found with ID {phase_id}"
        }

    # Fetch pipeline to get label config
    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}

    label_config = pipeline.label_configs[0]

    # Get dataset files for this phase
    dataset_files = await repos.DatasetFileRepository(db).get_by_phase(phase_id)

    if not dataset_files:
        return {
            "error": "No dataset file found",
            "message": "Dataset file not yet created for this phase. Generation may not have started."
        }

    # Use the first (most recent) dataset file
    ds_file = dataset_files[0]

    # Get all batch files for this dataset file
    batch_files = await repos.BatchGeneratedDatasetFileRepository(db).get_by_parent_file(ds_file.id)

    # Calculate current totals from batch files using stored sample_count
    # Note: Batch files append to the same temp file, so reading each file would give accumulated samples
    # Instead, we use the sample_count from the database which is correct for each batch
    total_samples_from_batches = 0
    label_counts = {}
    batch_files_info = []

    for batch_file in batch_files:
        # Use the sample_count from the database (which is correct for this specific batch)
        batch_sample_count = batch_file.sample_count
        total_samples_from_batches += batch_sample_count

        # Add batch file metadata only (no samples/examples)
        batch_data = batch_file.model_dump()
        # Explicitly exclude samples/examples fields if they exist
        batch_data.pop("samples", None)
        batch_data.pop("examples", None)
        batch_files_info.append(batch_data)

    # Build response
    ds_file_data = ds_file.model_dump()
    ds_file_data["current_sample_count"] = total_samples_from_batches
    ds_file_data["batch_count"] = len(batch_files)

    # Calculate label counts from final dataset file (if done) or from temp file (if in progress)
    # This gives accurate label distribution without loading all batch files
    file_to_read = None
    if ds_file.status == schemas.DATASET_FILE_STATUS.DONE and ds_file.file_path:
        file_to_read = ds_file.file_path
    elif batch_files:
        # Use the latest batch file (which has accumulated all samples so far)
        file_to_read = batch_files[-1].file_path

    if file_to_read:
        final_samples = []
        try:
            with open(file_to_read, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        final_samples.append(sample)

                        # Count labels
                        label = sample.get("label")
                        if isinstance(label, int):
                            label = str(label)
                        label_counts[label] = label_counts.get(label, 0) + 1

            # Only include samples in response if status is done
            if ds_file.status == schemas.DATASET_FILE_STATUS.DONE:
                ds_file_data["samples"] = final_samples
                logger.info(f"Loaded {len(final_samples)} final samples from completed dataset file")
            else:
                ds_file_data["samples"] = None  # Not done yet, don't include samples
                logger.info(f"Calculated label counts from {len(final_samples)} samples in progress")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read dataset file {file_to_read}: {e}")
            ds_file_data["samples"] = [] if ds_file.status == schemas.DATASET_FILE_STATUS.DONE else None
    else:
        ds_file_data["samples"] = None  # No file to read

    ds_file_data["label_counts"] = label_counts

    return {
        "message": f"Generation status retrieved: {len(batch_files)} batches, {total_samples_from_batches} samples",
        "data": {
            "phase_id": phase_id,
            "dataset_file": ds_file_data,
            "batch_files": batch_files_info,
        }
    }

@router.post("/phase/train/test")
async def test_train_model(
    payload: schemas.TestTrainPhaseRequest,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    phase = await repos.PhaseRepository(db).get_by_id(payload.phase_id)
    if not phase:
        return {"error": "Phase not found"}

    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}
    label_config = pipeline.label_configs[0]  # Get the first label config

    data_mgr = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.65,
        base_dir=f'{payload.cache_path}/test/{payload.phase_id}'
    )

    ds = data_mgr.to_datasets(file_path=payload.ds_file_path)

    trainer = services.TrainerService(
        base_model="prajjwal1/bert-tiny",
        label_config=label_config,
        base_dir=f'{payload.checkpoint_path}/test/{payload.phase_id}',
    )

    path, training_config = await trainer.train(
        ds,
        return_full_path=True,
        custom_training_config={
            "num_train_epochs": 5 
        }
    )

    return {
        "message": "Test training completed",
        "data": {
            "checkpoint_path": path,
            "training_config": training_config
        }
    }

@router.post("/evaluation/test")
async def test_evaluation(
    payload: schemas.TestEvaluationRequest,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    """Test endpoint for model evaluation without database mutations"""
    # Fetch pipeline and label config from database (read-only)
    pipeline = await repos.PipelineRepository(db).get_by_id(pipeline_id=payload.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}

    label_config = pipeline.label_configs[0]  # Get the first label config

    # Load frozen test set
    eval_ds = load_frozen_set(label_config, cache_path=payload.cache_path)

    # Create inference model
    inferencer = ProbModelInference(
        peft_path=payload.model_path,
        label_config=label_config,
    )

    # Evaluate the model
    eval_output = inferencer.evaluate(
        eval_ds,
        get_low_confidence=True,
        get_error_samples=True,
        confidence_threshold=0.5
    )

    return {
        "message": "Test evaluation completed",
        "data": {
            "model_path": payload.model_path,
            "cache_path": payload.cache_path,
            "overall_metrics": eval_output.get("overall", {}),
            "per_label_metrics": eval_output.get("per_label", {}),
            "error_samples_count": len(eval_output.get("error_samples", {}).get("samples", [])),
            "low_confidence_count": len(eval_output.get("low_confidence_correct", {}).get("samples", [])),
        }
    }

@router.post("/phase/{phase_id}/train")
async def train_model(
    phase_id: str,
    payload: Optional[schemas.StartTrainPhase] = None,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    '''
    1. Load training pool (composal dataset) from parent phase
    2. Optionally load training argument profile if provided
    3. Train model using TrainerService on the entire training pool
    4. TrainedRepository - save trained model info, ref current phase's DatasetFile
    5. Return trained model info
    '''
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {"error": "Phase not found"}

    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}
    label_config = pipeline.label_configs[0]  # Get the first label config

    # Fetch current phase's dataset file (for reference only)
    ds_file = (await repos.DatasetFileRepository(db).get_by_phase(phase_id))[0]
    if not ds_file:
        return {"error": "Dataset file not found for phase"}

    # Get the parent (root) phase's composal dataset for training pool
    parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
    if not parent_phase:
        # This IS the parent phase
        parent_phase = phase

    # Get composal dataset (training pool) from parent phase
    composal_dataset = await repos.DatasetRepository(db).get_by_phase(parent_phase.id)
    if not composal_dataset:
        return {"error": "No training pool (composal dataset) found for parent phase"}

    # Load training pool dataset for training
    training_pool_ds = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.65,
        absolute_dir=composal_dataset.file_path,
    ).to_datasets()

    # Load training profile if provided
    training_profile = None
    custom_training_config = None
    if payload and payload.training_argument_profile_id:
        training_profile = await repos.TrainingArgumentProfileRepository(db).get_by_id(
            payload.training_argument_profile_id
        )
        if not training_profile:
            return {
                "error": "Training argument profile not found",
                "message": f"No profile found with ID: {payload.training_argument_profile_id}"
            }
        # Merge training and LoRA configs for the trainer
        custom_training_config = training_profile.get_training_config()

    checkpoint_path, training_config = await services.TrainerService(
        base_model="prajjwal1/bert-tiny",
        label_config=label_config,
        base_dir=f'.checkpoints/{pipeline.id}/{phase.phase_number}'
    ).train(training_pool_ds, return_full_path=True, custom_training_config=custom_training_config)

    # Build description for FROM_SCRATCH training
    # Get all ancestor phases to build the composal dataset path
    ancestor_ids = phase.get_ancestor_phase_ids()
    if ancestor_ids:
        # Has parent phases
        ancestor_phases = []
        for ancestor_id in ancestor_ids:
            ancestor_phase = await repos.PhaseRepository(db).get_by_id(ancestor_id)
            if ancestor_phase:
                ancestor_phases.append(f"phase {ancestor_phase.phase_number}")
        phase_path_str = " -> ".join(ancestor_phases) + f" -> phase {phase.phase_number}"
        description = f"Trained from base model with composal dataset ({phase_path_str})"
    else:
        # Root phase
        description = f"Trained from base model with composal dataset (phase {phase.phase_number})"

    # Save trained model info - reference current phase's dataset file
    trained_info = await repos.TrainedModelRepository(db).create(
        phase_id=phase_id,
        model_name="Payment Classification Model v" + str(phase.phase_number),
        model_save_path=checkpoint_path,
        train_type="FROM_SCRATCH",
        dataset_file_id=ds_file.id,  # Reference to current phase's dataset file
        training_params=training_config,  # Save training configuration
        training_argument_profile_id=payload.training_argument_profile_id if payload else None,
        description=description,
        status="DONE",
    )

    return {
        "message": "Model trained successfully",
        "data": trained_info.model_dump(),
    }

@router.post("/phase/{phase_id}/continual-train")
async def continual_train_model(
    phase_id: str,
    payload: schemas.StartContinualTrainPhase,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    '''
    1. Load Datasets from current Phase and previous Phase
    2. Combine datasets with 3:7 ratio (previous:current)
    3. Continual train model using TrainerService from specified previous model checkpoint
    4. TrainedRepository - save trained model info, ref DatasetFile, Phase
    '''
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {"error": "Phase not found"}

    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}
    label_config = pipeline.label_configs[0]  # Get the first label config

    # Get the specified previous trained model
    prev_trained_model = await repos.TrainedModelRepository(db).get_by_id(payload.trained_model_id)
    if not prev_trained_model:
        return {
            "error": "Previous trained model not found",
            "message": f"No trained model found with ID: {payload.trained_model_id}"
        }

    # Get previous phase from the trained model
    prev_phase = await repos.PhaseRepository(db).get_by_id(prev_trained_model.phase_id)
    if not prev_phase:
        return {"error": "Previous phase not found"}

    ds_file = (await repos.DatasetFileRepository(db).get_by_phase(phase_id))[0]
    if not ds_file:
        return {"error": "Dataset file not found for phase"}

    prev_ds_file = (await repos.DatasetFileRepository(db).get_by_phase(prev_phase.id))[0]
    if not prev_ds_file:
        return {"error": "Dataset file not found for previous phase"}

    # Load datasets
    ds = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.65,
        absolute_dir=ds_file.file_path,
    ).to_datasets()
    prev_ds = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.65,
        absolute_dir=prev_ds_file.file_path,
    ).to_datasets()

    # Combine datasets with 3:7 ratio (previous:current)
    expect_old_data = min(round(ds_file.sample_count * 3/7), len(prev_ds))
    seed = random.randint(0, 1000)
    prev_ds_subset = prev_ds.shuffle(seed=seed).select(range(expect_old_data))
    combined_ds = concatenate_datasets([prev_ds_subset, ds])

    # Calculate actual ratio for description
    actual_prev_count = len(prev_ds_subset)
    actual_current_count = len(ds)
    total_count = actual_prev_count + actual_current_count
    # Calculate ratio as percentages
    prev_ratio = round((actual_prev_count / total_count) * 100) if total_count > 0 else 0
    current_ratio = round((actual_current_count / total_count) * 100) if total_count > 0 else 0

    # Build description for CONTINUAL training
    description = (f"Trained with phase {prev_phase.phase_number} dataset + "
                   f"phase {phase.phase_number} dataset with ratio: "
                   f"{prev_ratio}:{current_ratio} ({actual_prev_count}:{actual_current_count} samples)")

    # Load training profile if provided
    training_profile = None
    custom_training_config = None
    if payload.training_argument_profile_id:
        training_profile = await repos.TrainingArgumentProfileRepository(db).get_by_id(
            payload.training_argument_profile_id
        )
        if not training_profile:
            return {
                "error": "Training argument profile not found",
                "message": f"No profile found with ID: {payload.training_argument_profile_id}"
            }
        # Merge training and LoRA configs for the trainer
        custom_training_config = training_profile.get_training_config()

    # Train model using continual_train with previous checkpoint path
    checkpoint_path, training_config = await services.TrainerService(
        base_model="prajjwal1/bert-tiny",
        label_config=label_config,
        base_dir=f'.checkpoints/{pipeline.id}/{phase.phase_number}'
    ).continual_train(
        checkpoint_id=prev_trained_model.model_save_path,  # Pass full path to previous model
        dataset=combined_ds,
        return_full_path=True,
        custom_training_config=custom_training_config
    )

    trained_info = await repos.TrainedModelRepository(db).create(
        phase_id=phase_id,
        model_name=f"Payment Classification Model v{phase.phase_number} (Continual)",
        model_save_path=checkpoint_path,
        train_type="CONTINUAL",
        dataset_file_id=ds_file.id,
        training_params=training_config,  # Save training configuration
        training_argument_profile_id=payload.training_argument_profile_id,
        description=description,
        status="DONE",
    )

    return {
        "message": "Model continual trained successfully",
        "data": trained_info.model_dump(),
    }


@router.post("/phase/{phase_id}/evaluate")
async def evaluate_human_set(
    phase_id: str,
    payload: schemas.StartEvaluationPhase,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    '''
    1. Load frozen test set
    2. Load trained model by ID from payload
    3. Evaluate on frozen test set
    4. Evaluate on training pool for low confidence samples
    '''

    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {"error": "Phase not found"}
    # Fetch pipeline and label config from database
    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}
    label_config = pipeline.label_configs[0]  # Get the first label config

    # Fetch trained model by ID from payload
    trained_model_info = await repos.TrainedModelRepository(db).get_by_id(payload.trained_model_id)
    if not trained_model_info:
        return {"error": f"Trained model with ID {payload.trained_model_id} not found"}
    
    # Eval on frozen set
    eval_ds = load_frozen_set(label_config)
    inferencer = ProbModelInference(
        peft_path=trained_model_info.model_save_path,
        label_config=label_config,
    )

    eval_output = inferencer.evaluate(
        eval_ds,
        get_low_confidence=True,
        get_error_samples=True,
        confidence_threshold=payload.confidence_thresholds
    )

    # Eval on training pool for low confidence
    # Get the parent (root) phase's composal dataset for this sequence
    parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
    if not parent_phase:
        # This IS the parent phase
        parent_phase = phase

    base_ds = await repos.DatasetRepository(db).get_by_phase(parent_phase.id)
    training_ds = services.DataManager(label_config=label_config, absolute_dir=base_ds.file_path).to_datasets()
    training_ds = training_ds.shuffle(seed=random.randint(0, 1000))

    train_set_low_eval = inferencer.evaluate(
        training_ds,
        get_low_confidence=True,
        confidence_threshold=0.5,
        low_conf_limit=200
    )["low_confidence_correct"]["samples"]

    # Save evaluation results
    label_metrics = eval_output.get("per_label", {})
    label_metrics["recent_low_confidence_on_train"] = {
        "count": len(train_set_low_eval),
        "samples": train_set_low_eval
    }

    eval_info = await repos.EvaluationResultRepository(db).create(
        trained_model_id=trained_model_info.id,
        human_test_set_id=None,  # To be implemented
        dataset_file_id=trained_model_info.dataset_file_id,
        accuracy=eval_output["overall"]["accuracy"],
        precision=eval_output["overall"].get("macro_precision", 0.0),
        recall=eval_output["overall"].get("macro_recall", 0.0),
        f1_score=eval_output["overall"].get("macro_f1", 0.0),
        label_metrics=json.dumps(label_metrics),
    )

    # Update phase status to completed
    await repos.PhaseRepository(db).update_status(
        phase_id=phase_id,
        status=schemas.PHASE_STATUS.COMPLETED
    )

    resp = eval_info.model_dump()
    resp["label_metrics"] = json.loads(resp["label_metrics"])

    return {
        "message": "Evaluation completed",
        "data": resp,
    }

@router.post("/classify-error")
async def classify_errors(
    request: Request = None,
    db: AsyncSession = Depends(get_db_session),
    payload: schemas.ClassifyErrorRequest = None,
):
    '''
    1. Load error samples from latest evaluation
    2. Load error buckets from database
    3. Use ErrorCategorizer to categorize errors into buckets
    4. Save categorized errors into PhaseErrorBucket
    5. Return categorized error buckets
    6. Return error count map, examples map
    '''

    categorizer_llm = request.app.state.categorizer_llm

    prompt_mgr = request.app.state.prompt_mgr
    phase = await repos.PhaseRepository(db).get_by_id(payload.phase_id)
    if not phase:
        return {"error": "Phase not found"}
    # Fetch pipeline and label config from database
    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}
    label_config = pipeline.label_configs[0]  # Get the first label config

    # fetch trained model
    trained_model_info = (await repos.TrainedModelRepository(db).get_by_phase(payload.phase_id))[0]
    
    eval_ds = load_frozen_set(label_config)
    inferencer = ProbModelInference(
        peft_path=trained_model_info.model_save_path,
        label_config=label_config,
    )

    eval_output = inferencer.evaluate(
        eval_ds,
        get_low_confidence=True,
        get_error_samples=True,
        confidence_threshold=0.5
    )

    errors = eval_output["error_samples"]["samples"]

    # Categorize Error Buckets
    base_buckets = await repos.ErrorBucketRepository(db).list_by_pipeline(pipeline.id)

    bucket_map = {
        b.name: {
            "count": 0,
            "examples": [],
            "id": b.id,
        } for b in base_buckets
    }

    categorizer = services.ErrorCategorizer(
        label_config=label_config,
        llm=categorizer_llm,
        prompt_mgr=prompt_mgr,
        error_buckets=base_buckets,
    )

    err_testcases = [
        schemas.AnalyzeTestCase(
            sample=schemas.Sample(
                msg=err["text"],
                label=err["true_label"],
            ),
            predicted=err["predicted_label"],
            prob=err.get("probability", 0)
        ) for err in errors
    ]

    logger.info(f"Classifying {len(err_testcases)} error samples into buckets")

    categorized = await categorizer.batch_categorize_errors(err_testcases)

    for testcase, cat in categorized:
        if cat.bucket in bucket_map:
            bucket_map[cat.bucket]["count"] += 1
            if len(bucket_map[cat.bucket]["examples"]) < 100:
                bucket_map[cat.bucket]["examples"].append(testcase)

    # Make error counts map
    error_count = {
        bucket_map[k]["id"]: bucket_map[k]["count"] for k in bucket_map
    }

    examples_data = {
        bucket_map[k]["id"]: json.dumps([
            testcase.model_dump() for testcase in bucket_map[k]["examples"]
        ]) for k in bucket_map
    }
    # After categorized
    error_buckets = await repos.PhaseErrorBucketRepository(db).batch_create(
        phase_id=phase.id,
        buckets=base_buckets,
        error_counts=error_count,
        examples_data=examples_data,
    )

    return {
        "message": "Not implemented yet",
        "data": {
            "error_buckets": [b.model_dump() for b in error_buckets],
            "error_count": error_count,
            "examples_data": examples_data,
        }
    }

@router.get("/phase/{phase_id}/error-buckets")
async def view_error_buckets(
    phase_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """View error buckets for a given phase"""
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {"error": "Phase not found"}

    # Only get latest bucket for same bucket name (there might be duplicates due to retries)
    base_buckets = await repos.ErrorBucketRepository(db).list_by_pipeline(phase.pipeline_id)
    error_buckets = await repos.PhaseErrorBucketRepository(db).list_detail_by_phase(base_buckets, phase_id)

    return {
        "message": "Error buckets retrieved successfully",
        "data": error_buckets
    }   

@router.post("/phase/{phase_id}/error-buckets/generate")
async def generate_error_bucket_samples(
    phase_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Generate samples for error buckets in a given phase""" 
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {"error": "Phase not found"}

    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id) if phase else None
    label_config = pipeline.label_configs[0] if pipeline and pipeline.label_configs else None
    id2label = label_config.get_id2label() if label_config else {}
    
    base_buckets = await repos.ErrorBucketRepository(db).list_by_pipeline(phase.pipeline_id)
    error_buckets = await repos.PhaseErrorBucketRepository(db).list_detail_by_phase(base_buckets, phase_id)

    # Filter out buckets with 0 error count
    error_buckets = [b for b in error_buckets.values() if len(b["error"]["examples"]) > 0]

    if not error_buckets:
        return {
            "message": "No error found",
            "data": None
        }

    # Calculate total samples needed
    expect_error_fix = 300
    each_bucket_need = max(expect_error_fix // len(error_buckets), 50)
    error_gen_total = each_bucket_need * len(error_buckets)

    # Load latest evaluation result for this phase
    eval_repo = repos.EvaluationResultRepository(db)
    latest_eval = await eval_repo.get_latest_by_phase(phase_id)

    if not latest_eval:
        return {
            "message": "No evaluation found for this phase",
            "data": None
        }

    # Decode label_metrics to get low confidence examples
    label_metrics = latest_eval.get_label_metrics()
    all_low_confidence_samples = []
    if label_metrics and "recent_low_confidence_on_train" in label_metrics:
        low_conf_data = label_metrics["recent_low_confidence_on_train"]
        all_low_confidence_samples = low_conf_data.get("samples", [])

    expect_low_conf_each_label = error_gen_total // len(label_config.get_id2label().values())
    # Group low confidence examples by label and take equal quantity from each
    low_conf_by_label = {
        label: [] for label in id2label.values()
    }
    for sample in all_low_confidence_samples:
        label = sample.get("true_label")
        if sample.get("probability", 1.0) < 0.4 and len(low_conf_by_label[label]) < expect_low_conf_each_label:
            low_conf_by_label[label].append(sample)

    # Fill gaps with random samples from training set if needed
    # Get the parent (root) phase's composal dataset for this sequence
    parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
    if not parent_phase:
        # This IS the parent phase
        parent_phase = phase

    base_ds = await repos.DatasetRepository(db).get_by_phase(parent_phase.id)
    training_ds = services.DataManager(label_config=label_config, absolute_dir=base_ds.file_path).to_datasets()
    training_ds = training_ds.shuffle(seed=random.randint(0, 10000))

    for label in id2label.values():
        for item in training_ds:
            
            if len(low_conf_by_label[label]) >= expect_low_conf_each_label:
                break

            if id2label[str(item["label"])] == label:
                sample = {
                    "text": item['msg'],
                    "true_label": label,
                }
                low_conf_by_label[label].append(sample)
    
    print({k: len(v) for k, v in low_conf_by_label.items()})

    # Prepare generation config
    low_conf_stats = {
        label: len(samples) for label, samples in low_conf_by_label.items()
    }

    random_gen_total = error_gen_total // 2
    each_label_random_gen_total = random_gen_total // len(label_config.get_id2label())

    generation_config = {
        "random_each_label": each_label_random_gen_total,
        "low_confidence_each_label": expect_low_conf_each_label,
        "low_conf_stats": low_conf_stats,
        "each_error_bucket": each_bucket_need,
    }

    return {
        "message": "Low confidence samples loaded successfully",
        "data": {
            "error_buckets": error_buckets,
            "fix_generation_config": generation_config,
        }
    }

@router.post("/phase/{phase_id}/continue-gen")
async def continue_generation(
    phase_id: str,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    """
    Continue dataset generation from a parent phase

    1. Query the base/parent phase (root with phase_path="")
    2. Query all phases in sequence to find the previous/latest phase
    3. Get previous phase evaluation (for future use in targeted generation)
    4. Create new child phase as child of PREVIOUS phase and generate data
    5. Add generated data to parent's composal dataset
    """
    # 1. Query the prev phase and validate it's a root/parent phase
    prev_phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not prev_phase:
        return {"error": "Phase not found"}

    # Get pipeline and label config
    pipeline = await repos.PipelineRepository(db).get_by_id(prev_phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}

    label_config = pipeline.label_configs[0]

    # 2. Determine parent_phase (root) and previous_phase (latest in sequence)
    if prev_phase.phase_path == "":
        # Provided phase is the root/parent phase
        parent_phase =prev_phase 
    else:
        parent_phase = await repos.PhaseRepository(db).get_parent_phase(prev_phase.id)

    # 3. Validate parent phase has composal dataset
    parent_composal_ds = await repos.DatasetRepository(db).get_by_phase(parent_phase.id)
    if not parent_composal_ds:
        return {"error": "Parent phase has no composal dataset"}

    # Begin new phase creation
    new_phase = await repos.PhaseRepository(db).create(
        pipeline_id=pipeline.id,
        status=schemas.PHASE_STATUS.IN_PROGRESS,
        phase_number=prev_phase.phase_number + 1,
        previous_phase_id=prev_phase.id,  # Builds path by appending to previous phase
    )

    # Create a new dataset file under parent's composal dataset
    ds_file = await repos.DatasetFileRepository(db).create_dataset_file(
        parent_dataset_id=parent_composal_ds.id,
        file_path="",  # Will be updated after generation
        file_type="jsonl",
        phase_id=new_phase.id,
        sample_count=0  # Will be updated after generation
    )

    # Create DataManager and DataGeneratorV2 instances
    data_manager = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.65,
        base_dir=f'.cache/{pipeline.id}/{new_phase.id}/{new_phase.phase_number}'
    )

    # Create validator for label correction
    validator = services.DataValidator(
        llm=request.app.state.validator_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        label_config=label_config
    )

    data_generator_v2 = services.DataGeneratorV2(
        llm=request.app.state.teacher_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        data_manager=data_manager,
        high_llm=request.app.state.teacher_llm_high,
        validator=validator
    )

    # Load human seeds
    human_seeds = load_human_seed(label_config)

    # Create dict for label counts
    # TODO: Adjust based on evaluation results (targeted generation)
    label_count_dict = label_config.get_label2id()
    for key in label_count_dict:
        label_count_dict[key] = 200

    # Define callback to save each batch to database
    async def on_batch_generated(batch_number: int, samples: list, batch_file_path: str, metadata_path: str):
        """Callback to save batch information to database"""
        batch_repo = repos.BatchGeneratedDatasetFileRepository(db)
        await batch_repo.create_batch_file(
            parent_dataset_file_id=ds_file.id,
            file_path=batch_file_path,
            batch_number=batch_number,
            sample_count=len(samples)
        )
        logger.info(f"Saved batch {batch_number} with {len(samples)} samples to database (metadata: {metadata_path})")

    # Generate with batch tracking
    generated, path = await data_generator_v2.fresh_gen_v2(
        human_seeds=human_seeds,
        expect_total_each_label=label_count_dict,
        label_config=label_config,
        on_batch_generated=on_batch_generated,
        composal_ds_mgr=services.DataManager(
            label_config=label_config,
            rouge_threshold=0.65,
            absolute_dir=parent_composal_ds.file_path
        )
    )

    # Update DatasetFile with final path, sample count, and status
    await repos.DatasetFileRepository(db).update(
        file_id=ds_file.id,
        file_path=path,
        sample_count=len(generated),
        status=schemas.DATASET_FILE_STATUS.DONE
    )

    # Refresh to get updated data
    await db.refresh(ds_file)

    # Get all batch files for this dataset file
    batch_files = await repos.BatchGeneratedDatasetFileRepository(db).get_by_parent_file(ds_file.id)

    # Update parent composal dataset's file to include the new data
    # The parent composal dataset now has multiple dataset files
    # TODO: Optionally merge all dataset files into a single composal file

    return {
        "message": f"Generated {len(generated)} samples in {len(batch_files)} batches for new phase",
        "data": {
            "composal_ds": parent_composal_ds.model_dump(),
            "ds_file": ds_file.model_dump(),
            "batch_files": [bf.model_dump() for bf in batch_files],
        }
    }

# Information For UI
@router.get("/pipelines/{pipeline_id}")
async def pipeline_detail(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db_session),
    include_composal_datasets: bool = True,
    include_dataset_files: bool = True,
    include_error_buckets: bool = False
):
    '''
    Pipeline info with all nested relationships

    Query Parameters:
        include_composal_datasets: Include composal datasets for each phase (default: True)
        include_dataset_files: Include dataset files for each phase (default: True)
        include_error_buckets: Include error buckets for each phase (default: False)

    Response includes:
        1. Label Config
        2. Phases with nested relationships (based on query params):
           - ComposalDatasets
           - DatasetFiles
           - TrainedModels
           - EvaluationResults
    '''

    # Fetch pipeline with phases and requested relationships prefetched
    pipeline = await repos.PipelineRepository(db).get_with_phases(
        pipeline_id,
        include_phase_composal_datasets=include_composal_datasets,
        include_phase_dataset_files=include_dataset_files,
        include_phase_error_buckets=include_error_buckets
    )

    if not pipeline:
        return {"error": "Pipeline not found"}

    # Transform pipeline basic info
    pipeline_detail = pipeline.model_dump()

    # Process label config
    if pipeline.label_configs:
        pipeline_detail["label_config"] = pipeline.label_configs[0].model_dump()
        pipeline_detail["label_config"]["id2label"] = pipeline.label_configs[0].get_id2label()
        pipeline_detail["label_config"]["label2id"] = pipeline.label_configs[0].get_label2id()
        pipeline_detail["label_config"]["label_explanation"] = pipeline.label_configs[0].get_label_explanation()
    else:
        pipeline_detail["label_config"] = None

    # Process phases with all nested relationships
    phases_detail = []
    if pipeline.phases:
        for phase in pipeline.phases:
            phase_data = phase.model_dump()

            # Composal datasets and dataset files are already prefetched if requested
            if include_composal_datasets and hasattr(phase, 'composal_datasets'):
                phase_data["composal_datasets"] = [ds.model_dump() for ds in phase.composal_datasets]
            elif include_composal_datasets:
                # Fallback if not prefetched
                composal_dataset = await repos.DatasetRepository(db).get_by_phase(phase.id)
                phase_data["composal_datasets"] = [composal_dataset.model_dump()]

            if include_dataset_files and hasattr(phase, 'dataset_files'):
                phase_data["dataset_files"] = [df.model_dump() for df in phase.dataset_files]
            elif include_dataset_files:
                # Fallback if not prefetched
                dataset_files = await repos.DatasetFileRepository(db).get_by_phase(phase.id)
                phase_data["dataset_files"] = [df.model_dump() for df in dataset_files]

            if include_error_buckets and hasattr(phase, 'phase_error_buckets'):
                phase_data["phase_error_buckets"] = [peb.model_dump() for peb in phase.phase_error_buckets]

            # Fetch trained models for this phase (always included)
            trained_models = await repos.TrainedModelRepository(db).get_by_phase(phase.id)
            phase_data["trained_models"] = []

            for trained_model in trained_models:
                trained_model_data = trained_model.model_dump()

                # Parse training_params if exists
                if trained_model.training_params:
                    trained_model_data["training_params"] = trained_model.get_training_params()

                # Fetch training argument profile if exists
                if trained_model.training_argument_profile_id:
                    training_profile = await repos.TrainingArgumentProfileRepository(db).get_by_id(
                        trained_model.training_argument_profile_id
                    )
                    if training_profile:
                        trained_model_data["training_argument_profile"] = {
                            "id": training_profile.id,
                            "name": training_profile.name,
                            "description": training_profile.description,
                            "training_config": training_profile.get_training_config(),
                            "lora_config": training_profile.get_lora_config(),
                            "created_at": training_profile.created_at.isoformat(),
                            "updated_at": training_profile.updated_at.isoformat(),
                        }
                    else:
                        trained_model_data["training_argument_profile"] = None
                else:
                    trained_model_data["training_argument_profile"] = None

                # Fetch evaluation results for this trained model
                evaluation_results = await repos.EvaluationResultRepository(db).get_by_trained_model(trained_model.id)
                trained_model_data["evaluation_results"] = []

                for eval_result in evaluation_results:
                    eval_data = eval_result.model_dump()
                    # Parse JSON fields
                    if eval_result.label_metrics:
                        eval_data["label_metrics"] = eval_result.get_label_metrics()
                    if eval_result.metrics:
                        eval_data["metrics"] = eval_result.get_metrics()
                    trained_model_data["evaluation_results"].append(eval_data)

                phase_data["trained_models"].append(trained_model_data)

            phases_detail.append(phase_data)

    pipeline_detail["phases"] = phases_detail

    return {
        "data": pipeline_detail,
    }

@router.get("/pipelines/{pipeline_id}/testset")
async def get_testset(
    pipeline_id: str,
    db: AsyncSession = Depends(get_db_session),
    cache_path: str = ".cache"
):
    '''
    Get the loaded test set for a pipeline
    Returns data in a schema similar to DatasetFile model

    Args:
        pipeline_id: Pipeline ID (currently ignored, returns default frozen test set)
        cache_path: Path to cache directory containing frozen_test_set.json

    Returns:
        Test set data structured like DatasetFile model with samples
    '''
    # Load frozen test set from cache
    frozen_set_path = f"{cache_path}/frozen_test_set.json"

    try:
        with open(frozen_set_path, "r") as f:
            frozen_set = json.loads(f.read())

        # Get label distribution
        label_counts = {}
        for sample in frozen_set:
            label = sample.get("label")
            if isinstance(label, int):
                label = str(label)
            label_counts[label] = label_counts.get(label, 0) + 1

        # Return with DatasetFile-like schema
        from datetime import datetime
        return {
            "message": "Test set loaded successfully",
            "data": {
                "id": "frozen_test_set",  # Static ID for frozen test set
                "parent_dataset_id": None,  # No parent dataset
                "file_path": frozen_set_path,
                "phase_id": None,  # Not associated with a phase
                "file_type": "test",  # Type: test
                "sample_count": len(frozen_set),
                "created_at": datetime.now().isoformat(),  # Current timestamp
                "samples": frozen_set,  # Additional: actual sample data
                "label_counts": label_counts  # Additional: label distribution
            }
        }
    except FileNotFoundError:
        return {
            "error": f"Test set file not found at {frozen_set_path}",
            "message": "Please ensure the frozen test set exists in the cache directory"
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse test set JSON: {str(e)}",
            "message": "The test set file may be corrupted"
        }
    except Exception as e:
        return {
            "error": f"Failed to load test set: {str(e)}",
            "message": "An unexpected error occurred while loading the test set"
        }

@router.get("/phase/{phase_id}/trainingpool")
async def get_training_pool(
    phase_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    '''
    Get the training pool dataset for a phase
    Returns data in a schema similar to DatasetFile model

    Args:
        phase_id: Phase ID to fetch the training pool for

    Returns:
        Training pool data structured like DatasetFile model with samples
    '''
    # Fetch phase
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {
            "error": "Phase not found",
            "message": f"No phase found with ID {phase_id}"
        }

    # Get the parent (root) phase's composal dataset for this sequence
    # All phases in a sequence share the same composal dataset from the parent
    parent_phase = await repos.PhaseRepository(db).get_parent_phase(phase.id)
    if not parent_phase:
        # This IS the parent phase
        parent_phase = phase

    # Get composal datasets for the parent phase
    composal_dataset = await repos.DatasetRepository(db).get_by_phase(parent_phase.id)
    if not composal_dataset:
        return {
            "error": "No training pool found",
            "message": "No composal dataset exists for this phase sequence. Run first-gen to create one."
        }

    # Use the first (most recent) composal dataset for this phase
    training_pool_path = composal_dataset.file_path

    try:
        # Read the training pool file (JSONL format)
        samples = []
        with open(training_pool_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Get label distribution
        label_counts = {}
        for sample in samples:
            label = sample.get("label")
            if isinstance(label, int):
                label = str(label)
            label_counts[label] = label_counts.get(label, 0) + 1

        # Return with DatasetFile-like schema
        from datetime import datetime
        return {
            "message": "Training pool loaded successfully",
            "data": {
                "id": composal_dataset.id,
                "parent_dataset_id": composal_dataset.id,  # Reference to composal dataset
                "file_path": training_pool_path,
                "phase_id": phase_id,
                "file_type": "training_pool",
                "sample_count": len(samples),
                "created_at": composal_dataset.created_at.isoformat() if composal_dataset.created_at else datetime.now().isoformat(),
                "samples": samples,  # Additional: actual sample data
                "label_counts": label_counts  # Additional: label distribution
            }
        }
    except FileNotFoundError:
        return {
            "error": f"Training pool file not found at {training_pool_path}",
            "message": "The training pool file may have been deleted or moved"
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse training pool JSON: {str(e)}",
            "message": "The training pool file may be corrupted"
        }
    except Exception as e:
        return {
            "error": f"Failed to load training pool: {str(e)}",
            "message": "An unexpected error occurred while loading the training pool"
        }

@router.get("/phase/{phase_id}")
async def view_phase_detail(
    phase_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    '''
    Get detailed phase information with nested relationships

    Returns phase data with:
    - Composal datasets generated by this phase
    - Dataset files associated with this phase
    - Trained models for this phase
    - Evaluation results for trained models
    '''
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {"error": "Phase not found"}

    # Build phase data with nested relationships
    phase_data = phase.model_dump()

    # Get composal datasets for this phase
    composal_dataset = await repos.DatasetRepository(db).get_by_phase(phase.id)
    phase_data["composal_datasets"] = [composal_dataset.model_dump()] if composal_dataset else []

    # Get dataset files for this phase
    dataset_files = await repos.DatasetFileRepository(db).get_by_phase(phase.id)
    phase_data["dataset_files"] = [df.model_dump() for df in dataset_files] if dataset_files else []

    # Get trained models for this phase
    trained_models = await repos.TrainedModelRepository(db).get_by_phase(phase.id)
    phase_data["trained_models"] = []

    for trained_model in trained_models:
        trained_model_data = trained_model.model_dump()

        # Parse training_params if exists
        if trained_model.training_params:
            trained_model_data["training_params"] = trained_model.get_training_params()

        # Fetch training argument profile if exists
        if trained_model.training_argument_profile_id:
            training_profile = await repos.TrainingArgumentProfileRepository(db).get_by_id(
                trained_model.training_argument_profile_id
            )
            if training_profile:
                trained_model_data["training_argument_profile"] = {
                    "id": training_profile.id,
                    "name": training_profile.name,
                    "description": training_profile.description,
                    "training_config": training_profile.get_training_config(),
                    "lora_config": training_profile.get_lora_config(),
                    "created_at": training_profile.created_at.isoformat(),
                    "updated_at": training_profile.updated_at.isoformat(),
                }
            else:
                trained_model_data["training_argument_profile"] = None
        else:
            trained_model_data["training_argument_profile"] = None

        # Fetch evaluation results for this trained model
        evaluation_results = await repos.EvaluationResultRepository(db).get_by_trained_model(trained_model.id)
        trained_model_data["evaluation_results"] = []

        for eval_result in evaluation_results:
            eval_data = eval_result.model_dump()
            # Parse JSON fields
            if eval_result.label_metrics:
                eval_data["label_metrics"] = eval_result.get_label_metrics()
            if eval_result.metrics:
                eval_data["metrics"] = eval_result.get_metrics()
            trained_model_data["evaluation_results"].append(eval_data)

        phase_data["trained_models"].append(trained_model_data)

    # Child_phases_ids if current phase is initial phase
    if not phase.phase_path:
        child_phases = await repos.PhaseRepository(db).get_child_phases(phase.id)
        child_phases_dump = [cp.model_dump() for cp in child_phases] if child_phases else []
        phase_data["child_phases"] = child_phases_dump

    return {
        "message": "Phase details retrieved successfully",
        "data": phase_data
    }

# ===== Training Argument Profile Endpoints =====

@router.post("/training-profile")
async def create_training_profile(
    payload: schemas.CreateTrainingArgumentProfileRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new training argument profile"""
    profile_repo = repos.TrainingArgumentProfileRepository(db)

    try:
        # Serialize training and LoRA configs to JSON
        training_config_json = json.dumps(payload.training_config.model_dump(exclude_none=True))
        lora_config_json = json.dumps(payload.lora_config.model_dump(exclude_none=True))

        profile = await profile_repo.create(
            name=payload.name,
            training_config=training_config_json,
            lora_config=lora_config_json,
            description=payload.description
        )

        return {
            "message": "Training argument profile created successfully",
            "data": {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "training_config": profile.get_training_config(),
                "lora_config": profile.get_lora_config(),
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat(),
            }
        }
    except ValueError as e:
        return {
            "error": str(e),
            "message": "Failed to create training argument profile"
        }
    except Exception as e:
        logger.error(f"Unexpected error creating training profile: {str(e)}")
        return {
            "error": str(e),
            "message": "An unexpected error occurred"
        }

@router.get("/training-profiles")
async def list_training_profiles(
    db: AsyncSession = Depends(get_db_session),
):
    """List all training argument profiles"""
    profile_repo = repos.TrainingArgumentProfileRepository(db)
    profiles = await profile_repo.get_all()

    return {
        "message": "Training argument profiles retrieved successfully",
        "data": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "training_config": p.get_training_config(),
                "lora_config": p.get_lora_config(),
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
            }
            for p in profiles
        ]
    }

@router.get("/training-profile/{profile_id}")
async def get_training_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get a specific training argument profile by ID"""
    profile_repo = repos.TrainingArgumentProfileRepository(db)
    profile = await profile_repo.get_by_id(profile_id)

    if not profile:
        return {
            "error": "Training argument profile not found",
            "message": f"No profile found with ID: {profile_id}"
        }

    return {
        "message": "Training argument profile retrieved successfully",
        "data": {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "training_config": profile.get_training_config(),
            "lora_config": profile.get_lora_config(),
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }
    }

@router.get("/training-profile/name/{profile_name}")
async def get_training_profile_by_name(
    profile_name: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Get a specific training argument profile by name"""
    profile_repo = repos.TrainingArgumentProfileRepository(db)
    profile = await profile_repo.get_by_name(profile_name)

    if not profile:
        return {
            "error": "Training argument profile not found",
            "message": f"No profile found with name: {profile_name}"
        }

    return {
        "message": "Training argument profile retrieved successfully",
        "data": {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "training_config": profile.get_training_config(),
            "lora_config": profile.get_lora_config(),
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }
    }

@router.put("/training-profile/{profile_id}")
async def update_training_profile(
    profile_id: str,
    payload: schemas.UpdateTrainingArgumentProfileRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Update an existing training argument profile"""
    profile_repo = repos.TrainingArgumentProfileRepository(db)

    try:
        # Serialize configs if provided
        training_config_json = None
        lora_config_json = None

        if payload.training_config:
            training_config_json = json.dumps(payload.training_config.model_dump(exclude_none=True))
        if payload.lora_config:
            lora_config_json = json.dumps(payload.lora_config.model_dump(exclude_none=True))

        profile = await profile_repo.update(
            profile_id=profile_id,
            name=payload.name,
            description=payload.description,
            training_config=training_config_json,
            lora_config=lora_config_json
        )

        if not profile:
            return {
                "error": "Training argument profile not found",
                "message": f"No profile found with ID: {profile_id}"
            }

        return {
            "message": "Training argument profile updated successfully",
            "data": {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "training_config": profile.get_training_config(),
                "lora_config": profile.get_lora_config(),
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat(),
            }
        }
    except ValueError as e:
        return {
            "error": str(e),
            "message": "Failed to update training argument profile"
        }
    except Exception as e:
        logger.error(f"Unexpected error updating training profile: {str(e)}")
        return {
            "error": str(e),
            "message": "An unexpected error occurred"
        }

@router.delete("/training-profile/{profile_id}")
async def delete_training_profile(
    profile_id: str,
    force: bool = False,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Delete a training argument profile.

    This endpoint:
    1. Checks if any trained models are using this profile
    2. If models exist and force=False: Returns error
    3. If models exist and force=True: Sets their profile_id to NULL, then deletes
    4. Deletes the profile from database

    Args:
        profile_id: ID of the profile to delete
        force: Force deletion even if models are using this profile (default: False)
        db: Database session

    Returns:
        DeleteTrainingProfileResponse with deletion status

    Example:
        DELETE /v2/workflow/training-profile/abc123?force=true
    """
    try:
        profile_repo = repos.TrainingArgumentProfileRepository(db)
        trained_model_repo = repos.TrainedModelRepository(db)

        logger.info(f"Deleting training profile: {profile_id}")

        # Get the profile
        profile = await profile_repo.get_by_id(profile_id)

        if not profile:
            return {
                "error": "Training argument profile not found",
                "message": f"No profile found with ID: {profile_id}"
            }

        profile_name = profile.name

        # Check if any trained models are using this profile
        from sqlmodel import select
        from app.core.models.models import TrainedModel

        statement = select(TrainedModel).where(
            TrainedModel.training_argument_profile_id == profile_id
        )
        result = await db.execute(statement)
        models_using_profile = list(result.scalars().all())
        models_count = len(models_using_profile)

        logger.info(f"Found {models_count} trained models using this profile")

        # If models are using this profile and force is False, return error
        if models_count > 0 and not force:
            return {
                "error": "Profile in use",
                "message": f"Cannot delete profile '{profile_name}'. "
                          f"{models_count} trained model(s) are using this profile. "
                          f"Use force=true to delete anyway (will set model profile_id to NULL).",
                "models_count": models_count
            }

        # If force=True and models exist, set their profile_id to NULL
        models_updated = 0
        if models_count > 0 and force:
            logger.info(f"Force deletion: Clearing profile reference from {models_count} models")

            for model in models_using_profile:
                model.training_argument_profile_id = None
                db.add(model)
                models_updated += 1

            await db.commit()
            logger.info(f"Cleared profile reference from {models_updated} models")

        # Delete the profile
        deleted = await profile_repo.delete(profile_id)

        if not deleted:
            return {
                "error": "Deletion failed",
                "message": f"Failed to delete profile: {profile_id}"
            }

        logger.info(f"Successfully deleted training profile: {profile_id}")

        return {
            "message": "Training argument profile deleted successfully",
            "profile_id": profile_id,
            "profile_name": profile_name,
            "database_deleted": True,
            "models_updated": models_updated
        }

    except Exception as e:
        logger.error(f"Unexpected error deleting training profile: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "message": "An unexpected error occurred"
        }

# ===== Model Inference Endpoint =====

@router.post("/inference")
async def inference_model(
    payload: schemas.ModelInferenceRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Run inference on trained model with input texts

    Args:
        model_path: Path to the trained model checkpoint
        texts: List of texts to classify
        pipeline_id: Pipeline ID to get label configuration

    Returns:
        Predictions for each input text with labels and probabilities
    """
    try:
        # Get pipeline and label config
        pipeline = await repos.PipelineRepository(db).get_by_id(payload.pipeline_id)
        if not pipeline or not pipeline.label_configs:
            return {
                "error": "Pipeline or label config not found",
                "message": f"No pipeline found with ID: {payload.pipeline_id}"
            }

        label_config = pipeline.label_configs[0]

        # Verify model path exists
        from pathlib import Path
        model_path = Path(payload.model_path)
        if not model_path.exists():
            return {
                "error": "Model path not found",
                "message": f"The model path does not exist: {payload.model_path}"
            }

        # Initialize inference model
        logger.info(f"Loading model from {payload.model_path} for inference")
        inference_model = ProbModelInference(
            peft_path=payload.model_path,
            label_config=label_config
        )

        # Run predictions
        predictions = inference_model.predict(payload.texts)

        # Format results
        formatted_predictions = []
        for text, pred in zip(payload.texts, predictions):
            formatted_predictions.append({
                "text": text,
                "label": pred["label"],
                "probability": pred["prob"],
                "all_probabilities": pred["all_probs"]
            })

        return {
            "message": "Inference completed successfully",
            "predictions": formatted_predictions,
            "model_info": {
                "model_path": payload.model_path,
                "labels": label_config.get_id2label(),
                "total_predictions": len(formatted_predictions)
            }
        }

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return {
            "error": str(e),
            "message": "An error occurred during inference"
        }


@router.post("/convert-to-onnx")
async def convert_to_onnx(
    payload: schemas.ConvertToONNXRequest,
    request: Request,
):
    """
    Convert a trained model to ONNX format for optimized inference.

    This endpoint:
    1. Loads the merged model from {model_path}/_merged
    2. Loads the tokenizer from {model_path}
    3. Converts the model to ONNX format
    4. Saves both ONNX model and tokenizer to a new directory
    5. Creates a zip file for download

    Args:
        payload: Contains model_path and optional output_name
        request: FastAPI request object

    Returns:
        FileResponse with the zipped ONNX model

    Example:
        POST /v2/workflow/convert-to-onnx
        {
            "model_path": ".checkpoints/pipeline_id/phase_number",
            "output_name": "my_onnx_model"  # optional
        }
    """
    try:
        from pathlib import Path
        import shutil
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        from fastapi.responses import FileResponse
        import os

        logger.info(f"Converting model to ONNX: {payload.model_path}")

        # Validate model path
        model_path = Path(payload.model_path)
        if not model_path.exists():
            return {
                "error": "Model path not found",
                "message": f"The model path does not exist: {payload.model_path}"
            }

        # Check for _merged folder
        merged_path = model_path / "_merged"
        if not merged_path.exists():
            return {
                "error": "Merged model not found",
                "message": f"The _merged folder does not exist in: {payload.model_path}"
            }

        # Determine output name
        output_name = payload.output_name or model_path.name
        temp_output_dir = Path(".cache/onnx") / f"{output_name}_temp"

        # Clean up existing temp directory if it exists
        if temp_output_dir.exists():
            logger.info(f"Removing existing temp directory: {temp_output_dir}")
            shutil.rmtree(temp_output_dir)

        # Create directory structure:
        # {output_name}_temp/
        #   ├── tokenizer files (root level)
        #   └── onnx/
        #       └── model.onnx
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        onnx_subdir = temp_output_dir / "onnx"
        onnx_subdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model from {merged_path} for ONNX conversion")

        # Load and convert model to ONNX
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(merged_path),
            export=True,
        )

        # Load tokenizer from the model path (not _merged)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Save tokenizer to root of temp directory
        logger.info(f"Saving tokenizer to {temp_output_dir}")
        tokenizer.save_pretrained(str(temp_output_dir))

        # Save ONNX model to onnx/ subdirectory
        logger.info(f"Saving ONNX model to {onnx_subdir}")
        ort_model.save_pretrained(str(onnx_subdir))

        # Create zip file
        zip_path = Path(".cache/onnx") / f"{output_name}.zip"
        logger.info(f"Creating zip file: {zip_path}")

        # Remove existing zip if present
        if zip_path.exists():
            os.remove(zip_path)

        # Create zip archive from temp directory
        shutil.make_archive(
            str(zip_path.with_suffix('')),  # base name without extension
            'zip',  # archive format
            str(temp_output_dir)  # directory to archive
        )

        logger.info(f"ONNX conversion completed successfully")

        # Return the zip file for download
        return FileResponse(
            path=str(zip_path),
            media_type='application/zip',
            filename=f"{output_name}.zip",
            headers={
                "Content-Disposition": f"attachment; filename={output_name}.zip"
            }
        )

    except Exception as e:
        logger.error(f"ONNX conversion error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "message": "An error occurred during ONNX conversion"
        }


@router.post("/delete-model")
async def delete_trained_model(
    payload: schemas.DeleteModelRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Delete a trained model from database and optionally from disk.

    This endpoint:
    1. Retrieves the trained model from database by ID
    2. Optionally deletes physical model files from disk
    3. Deletes the model record from database
    4. Returns deletion status

    Args:
        payload: Contains model_id and delete_files flag
        db: Database session

    Returns:
        DeleteModelResponse with deletion status

    Example:
        POST /v2/workflow/delete-model
        {
            "model_id": "abc123-model-id",
            "delete_files": true
        }
    """
    try:
        from pathlib import Path
        import shutil

        logger.info(f"Deleting trained model: {payload.model_id}")

        # Get trained model repository
        trained_model_repo = repos.TrainedModelRepository(db)

        # Retrieve the model
        trained_model = await trained_model_repo.get_by_id(payload.model_id)

        if not trained_model:
            return {
                "error": "Model not found",
                "message": f"No trained model found with ID: {payload.model_id}"
            }

        model_path = trained_model.model_save_path
        files_deleted = False
        phase_checkpoint_cleared = False

        # Delete physical files if requested
        if payload.delete_files and model_path:
            model_path_obj = Path(model_path)

            if model_path_obj.exists():
                logger.info(f"Deleting model files at: {model_path}")

                try:
                    # If it's a directory, delete recursively
                    if model_path_obj.is_dir():
                        shutil.rmtree(model_path_obj)
                        logger.info(f"Deleted model directory: {model_path}")
                    # If it's a file, delete the file
                    elif model_path_obj.is_file():
                        model_path_obj.unlink()
                        logger.info(f"Deleted model file: {model_path}")

                    files_deleted = True
                except Exception as e:
                    logger.error(f"Error deleting model files: {e}")
                    return {
                        "error": "File deletion failed",
                        "message": f"Failed to delete model files at {model_path}: {str(e)}",
                        "model_id": payload.model_id,
                        "files_deleted": False,
                        "database_deleted": False
                    }
            else:
                logger.warning(f"Model path does not exist: {model_path}")
                # Continue with database deletion even if files don't exist

        # Check if this model is referenced in any phase's checkpoint
        # If so, we should clear the phase's checkpoint_path and checkpoint_id
        phase_repo = repos.PhaseRepository(db)
        phase = await phase_repo.get_by_id(trained_model.phase_id)

        if phase and phase.checkpoint_path == model_path:
            # This model is the phase's checkpoint - clear it
            logger.info(f"Clearing phase checkpoint reference for phase: {phase.id}")
            await phase_repo.update(
                phase.id,
                checkpoint_id=None,
                checkpoint_path=None
            )
            await db.commit()
            phase_checkpoint_cleared = True

        # Delete from database
        database_deleted = await trained_model_repo.delete(payload.model_id)

        if not database_deleted:
            return {
                "error": "Database deletion failed",
                "message": f"Failed to delete model from database: {payload.model_id}"
            }

        logger.info(f"Successfully deleted trained model: {payload.model_id}")

        return {
            "message": "Model deleted successfully",
            "model_id": payload.model_id,
            "model_path": model_path,
            "files_deleted": files_deleted,
            "database_deleted": database_deleted,
            "phase_checkpoint_cleared": phase_checkpoint_cleared
        }

    except Exception as e:
        logger.error(f"Model deletion error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "message": "An error occurred during model deletion"
        }