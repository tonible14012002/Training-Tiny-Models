from fastapi import APIRouter, Request, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core import repositories as repos
from app.api.dependencies import get_db_session
from app.core import services
from app.core import schemas
from src.payment_classifier.inference.prob_inference import ProbModelInference
from app.utils.dataset_helper import DatasetHelper
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

    # Create new iteration phase
    phase = await repos.PhaseRepository(db).create(
        pipeline_id=pipeline.id,
        status=schemas.PHASE_STATUS.IN_PROGRESS,
        phase_number=0,
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
        rouge_threshold=0.6,
        base_dir=f'.cache/{pipeline.id}/{phase.id}/{phase.phase_number}'
    )

    data_generator_v2 = services.DataGeneratorV2(
        llm=request.app.state.teacher_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        data_manager=data_manager,
    )

    human_seeds = load_human_seed(label_config)

    # Create dict
    label_count_dict = label_config.get_label2id()
    for key in label_count_dict:
        label_count_dict[key] = 200

    # Define callback to save each batch to database
    async def on_batch_generated(batch_number: int, samples: list, temp_file_path: str):
        """Callback to save batch information to database"""
        batch_repo = repos.BatchGeneratedDatasetFileRepository(db)
        await batch_repo.create_batch_file(
            parent_dataset_file_id=ds_file.id,
            file_path=temp_file_path,
            batch_number=batch_number,
            sample_count=len(samples)
        )
        logger.info(f"Saved batch {batch_number} with {len(samples)} samples to database")

    # Generate with batch tracking
    generated, path = await data_generator_v2.fresh_gen_v2(
        human_seeds=human_seeds,
        expect_total_each_label=label_count_dict,
        on_batch_generated=on_batch_generated
    )

    # Copy final file to composal directory
    source_path = Path(path)
    shutil.copy2(source_path, composal_file_path)

    # Update DatasetFile with final path and sample count
    await repos.DatasetFileRepository(db).update(
        file_id=ds_file.id,
        file_path=path,
        sample_count=len(generated)
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
        rouge_threshold=0.6,
        base_dir=f'{payload.cache_path}/test/first-gen/{payload.pipeline_id}'
    )

    data_generator_v2 = services.DataGeneratorV2(
        llm=request.app.state.teacher_llm,
        prompt_mgr=request.app.state.prompt_mgr,
        data_manager=data_manager,
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
        expect_total_each_label=label_count_dict
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
        rouge_threshold=0.6,
        base_dir=f'{payload.cache_path}/test/{payload.phase_id}'
    )

    ds = data_mgr.to_datasets(file_path=payload.ds_file_path)

    trainer = services.TrainerService(
        base_model="prajjwal1/bert-tiny",
        label_config=label_config,
        base_dir=f'{payload.checkpoint_path}/test/{payload.phase_id}',
    )

    path = await trainer.train(ds, return_full_path=True)

    return {
        "message": "Test training completed",
        "data": {
            "checkpoint_path": path
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
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    '''
    1. Load Datasets from generation Phase
    2. Train model using TrainerService
    3. TrainedRepository - save trained model info, ref DatasetFile, Phase
    4. Return trained model info
    '''
    phase = await repos.PhaseRepository(db).get_by_id(phase_id)
    if not phase:
        return {"error": "Phase not found"}

    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}
    label_config = pipeline.label_configs[0]  # Get the first label config

    # Fetch ds_file for phase
    ds_file = (await repos.DatasetFileRepository(db).get_by_phase(phase_id))[0]
    if not ds_file:
        return {"error": "Dataset file not found for phase"}

    # Create DataManager instance for this pipeline's label config
    data_manager = services.DataManager(
        label_config=label_config,
        rouge_threshold=0.6,
        absolute_dir=ds_file.file_path,
    )
    ds = data_manager.to_datasets()

    checkpoint_path = await services.TrainerService(
        base_model="prajjwal1/bert-tiny",
        label_config=label_config,
        base_dir=f'.checkpoints/{pipeline.id}/{phase.phase_number}'
    ).train(ds, return_full_path=True)

    # Save trained model info
    trained_info = await repos.TrainedModelRepository(db).create(
        phase_id=phase_id,
        model_name="Payment Classification Model v" + str(phase.phase_number),
        model_save_path=checkpoint_path,
        dataset_file_id=ds_file.id,
        status="DONE",
    )

    return {
        "message": "Model trained successfully",
        "data": trained_info.model_dump(),
    }

@router.post("/phase/evaluate")
async def evaluate_human_set(
    payload: schemas.StartEvaluationPhase,
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    '''
    1. Load frozen test set
    2. Load trained model for the phase
    3. Evaluate on frozen test set
    4. Evaluate on training pool for low confidence samples
    '''

    phase = await repos.PhaseRepository(db).get_by_id(payload.phase_id)
    if not phase:
        return {"error": "Phase not found"}
    # Fetch pipeline and label config from database
    pipeline = await repos.PipelineRepository(db).get_by_id(phase.pipeline_id)
    if not pipeline or not pipeline.label_configs:
        return {"error": "Pipeline or label config not found"}
    label_config = pipeline.label_configs[0]  # Get the first label config

    trained_model_info = (await repos.TrainedModelRepository(db).get_by_phase(payload.phase_id))[0]
    
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
    base_ds = (await repos.DatasetRepository(db).get_by_pipeline(pipeline.id))[0]
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

    resp = eval_info.model_dump()
    resp["label_metrics"] = json.loads(resp["label_metrics"])

    # Fetch previous phase evaluation
    # phases = await repos.PhaseRepository(db).get_by_pipeline(pipeline.id)
    # prev_phases = [p for p in phases if p.phase_number < phase.phase_number]

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
    base_ds = (await repos.DatasetRepository(db).get_by_pipeline(phase.pipeline_id))[0]
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

# UI
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
                composal_datasets = await repos.DatasetRepository(db).get_by_phase(phase.id)
                phase_data["composal_datasets"] = [ds.model_dump() for ds in composal_datasets]

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

    # Get composal datasets for this phase
    composal_datasets = await repos.DatasetRepository(db).get_by_phase(phase_id)
    if not composal_datasets:
        return {
            "error": "No training pool found",
            "message": "No composal dataset exists for this phase. Run first-gen to create one."
        }

    # Use the first (most recent) composal dataset for this phase
    base_ds = composal_datasets[0]
    training_pool_path = base_ds.file_path

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
                "id": base_ds.id,
                "parent_dataset_id": base_ds.id,  # Reference to composal dataset
                "file_path": training_pool_path,
                "phase_id": phase_id,
                "file_type": "training_pool",
                "sample_count": len(samples),
                "created_at": base_ds.created_at.isoformat() if base_ds.created_at else datetime.now().isoformat(),
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
    composal_datasets = await repos.DatasetRepository(db).get_by_phase(phase.id)
    phase_data["composal_datasets"] = [ds.model_dump() for ds in composal_datasets] if composal_datasets else []

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