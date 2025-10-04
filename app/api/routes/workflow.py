"""Workflow endpoints."""

from fastapi import APIRouter, Request, Query, Body
from app.core import services
from app.core import schemas
from app.core.schemas.workflow import PAYMENT_LABEL_V2
from app.core.schemas.inference import ThresholdConfig
from src.payment_classifier.inference import ADBModelInference
from src.payment_classifier.inference.prob_inference import ProbModelInference

import json
import logging
from pydantic import TypeAdapter
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflow", tags=["workflow"])


def _detect_inference_type(checkpoint_path: str) -> str:
    """Detect inference type from checkpoint metadata, fallback to ADB if not found"""
    config_file = Path(checkpoint_path) / "inference_config.json"

    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get("inference_type", "prob")
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Could not read inference config from {config_file}, defaulting to prob")

    # Fallback: check if ADB data exists
    adb_file = Path(checkpoint_path) / "adb_data.json"
    if adb_file.exists():
        return "adb"

    # Default to prob for backward compatibility
    return "prob"

@router.post("/generate-data-v2")
async def generate_synthetic_data_v2(request: Request):
    """Generate synthetic data using DataGeneratorV2 with parallel processing."""
    data_generator_v2: services.DataGeneratorV2 = request.app.state.data_generator_v2
    path = ".cache/human_seed.json"
    # load text from json file
    with open(path, "r") as f:
        seed_data = json.load(f)
    logger.info(f"Loaded {len(seed_data)} seed examples from {path}")
    converted = [
        {**seed, 'label': PAYMENT_LABEL_V2.to_str(seed['label'])}
        for seed in seed_data
    ]

    human_seeds = TypeAdapter(List[schemas.Sample]).validate_python(converted)
    await data_generator_v2.fresh_gen(human_seeds)

    return {"message": "Data generation v2 started with parallel processing", "status": "in_progress"}

@router.post("/fresh-generate-eval")
async def generate_fresh_eval_data(request: Request):
    """Generate fresh evaluation data using eval generator - both intent and open intent."""
    eval_generator: services.EvalGenerator = request.app.state.eval_generator
    # path = ".cache/human_seed.json"

    # # Load human seed data for intent generation
    # with open(path, "r") as f:
    #     seed_data = json.load(f)
    # logger.info(f"Loaded {len(seed_data)} seed examples from {path}")

    # converted = [
    #     {**seed, 'label': PAYMENT_LABEL_V2.to_str(seed['label'])}
    #     for seed in seed_data
    # ]

    human_seeds = []
    # human_seeds = TypeAdapter(List[schemas.Sample]).validate_python(converted)

    # Optional: Load seed messages for open intent (can be empty for now)
    human_seed_messages = []  # Can be extended to load from a file

    # Generate fresh evaluation data (both intent and open intent)
    result = await eval_generator.fresh_gen(human_seeds, human_seed_messages)

    return {
        "message": "Evaluation data generation completed",
        "status": "completed",
        "iteration_number": result["iteration_number"],
        "intent_samples_generated": result["intent_count"],
        "open_intent_messages_generated": result["open_intent_count"],
        "total_generated": result["total_generated"]
    }

@router.post("/train")
async def train_student_model(request: Request, inference_type: str = "prob"):
    """Train model with specified inference type (adb or prob)"""
    data_manager: services.DataManager = request.app.state.data_manager
    trainer_service: services.TrainerService = request.app.state.trainer_service

    if inference_type not in ["adb", "prob"]:
        return {
            "status": "error",
            "message": "inference_type must be 'adb' or 'prob'"
        }

    ds = data_manager.to_datasets()
    checkpoint_num = await trainer_service.train(ds, inference_type=inference_type)

    return {
        "status": "completed",
        "checkpoint_number": checkpoint_num,
        "inference_type": inference_type
    }

@router.post("/continual-train")
async def continual_train_model(
    request: Request,
    checkpoint_id: str,
    inference_type: str = "prob",
    dataset_path: str = None
):
    """Continue training from an existing checkpoint with sub-versioning.

    Args:
        checkpoint_id: The checkpoint identifier to continue from (e.g., "10", "10.1", "11.2")
        inference_type: Type of inference ("adb" or "prob")
        dataset_path: Optional path to specific dataset file. If None, uses all accumulated data

    Returns:
        Response with new sub-checkpoint identifier (e.g., "10.1", "10.2")
    """
    data_manager: services.DataManager = request.app.state.data_manager
    trainer_service: services.TrainerService = request.app.state.trainer_service

    if inference_type not in ["adb", "prob"]:
        return {
            "status": "error",
            "message": "inference_type must be 'adb' or 'prob'"
        }

    # Get checkpoint path and validate it exists
    checkpoint_path = trainer_service.get_item_path_by_id(checkpoint_id)
    if checkpoint_path is None:
        return {
            "status": "error",
            "message": f"Checkpoint {checkpoint_id} does not exist"
        }

    # Validate dataset_path if provided
    if dataset_path and not Path(dataset_path).exists():
        return {
            "status": "error",
            "message": f"Dataset file not found: {dataset_path}"
        }

    try:
        # Load dataset from specific path or all data
        ds = data_manager.to_datasets(file_path=dataset_path)

        # Continue training from the specified checkpoint (supports both "10" and "10.1" formats)
        new_checkpoint_id = await trainer_service.continual_train(
            checkpoint_id=checkpoint_id,
            dataset=ds,
            inference_type=inference_type
        )

        return {
            "status": "completed",
            "checkpoint_id": new_checkpoint_id,
            "base_checkpoint": checkpoint_id,
            "inference_type": inference_type,
            "dataset_path": dataset_path or data_manager.LOCAL_FILE,
            "message": f"Continual training completed. New checkpoint: {new_checkpoint_id}"
        }
    except Exception as e:
        logger.error(f"Continual training failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Continual training failed: {str(e)}"
        }

@router.post("/evaluate")
async def evaluate_model(
    request: Request,
    payload: schemas.EvaluationRequest = schemas.EvaluationRequest()
) -> schemas.EvaluationResponse:
    """Evaluate model performance using the latest evaluation dataset with comprehensive analysis.

    Args:
        payload.checkpoint_id: Optional checkpoint identifier (e.g., "1", "2", "1.1", "2.3").
                               If not provided, uses the latest checkpoint.
    """

    # Get services from app state
    model_analyzer: services.ModelAnalyzer = request.app.state.model_analyzer
    trainer_service: services.TrainerService = request.app.state.trainer_service
    eval_data_manager: services.EvalDataManager = request.app.state.eval_data_manager

    try:
        # Get checkpoint path - use specific checkpoint if provided, otherwise latest
        if payload.checkpoint_id:
            checkpoint_path = trainer_service.get_item_path_by_id(payload.checkpoint_id)
            if checkpoint_path is None:
                return schemas.EvaluationResponse(
                    message=f"Checkpoint {payload.checkpoint_id} not found",
                    status="error",
                    checkpoint_path="",
                    checkpoint_id=payload.checkpoint_id,
                    evaluation_data_info={},
                    results={}
                )
            checkpoint_id = payload.checkpoint_id
            logger.info(f"Using specified checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = trainer_service.get_latest_item_path()
            if checkpoint_path is None:
                return schemas.EvaluationResponse(
                    message="No trained model checkpoint found",
                    status="error",
                    checkpoint_path="",
                    checkpoint_id=None,
                    evaluation_data_info={},
                    results={}
                )
            checkpoint_id = Path(checkpoint_path).name
            logger.info(f"Using latest checkpoint: {checkpoint_path}")

        # Load evaluation dataset
        eval_samples = eval_data_manager.load(payload.iteration_number)
        if not eval_samples:
            return schemas.EvaluationResponse(
                message="No evaluation data found",
                status="error",
                checkpoint_path=str(checkpoint_path),
                checkpoint_id=checkpoint_id,
                evaluation_data_info={},
                results={}
            )

        logger.info(f"Loaded {len(eval_samples)} evaluation samples")

        # Convert to dataset format using EvalDataManager
        eval_dataset = eval_data_manager.to_datasets(payload.iteration_number)

        # Load open intent data if requested
        open_intent_samples = None
        if payload.include_open_intent:
            open_intent_samples = eval_data_manager.load_open_intent(payload.iteration_number)
            logger.info(f"Loaded {len(open_intent_samples) if open_intent_samples else 0} open intent samples")

        # Parse threshold config if provided
        threshold_config = None
        if payload.threshold_config:
            threshold_config = ThresholdConfig(**payload.threshold_config)
            logger.info(f"Using threshold config: {threshold_config}")

        # Load model and run comprehensive evaluation
        # For PAYMENT_LABEL_V2, always use probability-based inference
        prob_inferencer = ProbModelInference(
            peft_path=str(checkpoint_path),
            label_config=PAYMENT_LABEL_V2,
            threshold_config=threshold_config
        )
        model_analyzer.load_model(str(checkpoint_path), inferencer=prob_inferencer)

        evaluation_result = model_analyzer.analyze_model(
            evaluation_dataset=eval_dataset,
            open_intent_samples=open_intent_samples,
            include_test_cases=payload.include_test_cases,
        )

        # # Generate analysis report
        error_patterns = model_analyzer.get_error_patterns_from_result(evaluation_result.errors_by_label) if evaluation_result.errors_by_label else {}

        # Prepare evaluation data info
        iteration_num = payload.iteration_number or eval_data_manager.get_latest_item_number()
        eval_info = {
            "iteration_number": iteration_num,
            "known_intent_samples": len(eval_samples),
            "open_intent_samples": len(open_intent_samples) if open_intent_samples else 0,
            "total_samples": len(eval_samples) + (len(open_intent_samples) if open_intent_samples else 0)
        }

        # Prepare results
        results = {
            "overall_metrics": evaluation_result.overall.model_dump(),
            "per_label_metrics": {k: v.model_dump() for k, v in evaluation_result.per_label.items()},
            "open_intent_analysis": evaluation_result.open_intent_analysis.model_dump() if evaluation_result.open_intent_analysis else None,
            "adb_info": evaluation_result.adb_info,
            "error_patterns": {k: [case.model_dump() for case in cases] for k, cases in error_patterns.items()},
        }

        return schemas.EvaluationResponse(
            message="Model evaluation completed successfully",
            status="completed",
            checkpoint_path=str(checkpoint_path),
            checkpoint_id=checkpoint_id,
            evaluation_data_info=eval_info,
            results=results
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return schemas.EvaluationResponse(
            message=f"Evaluation failed: {str(e)}",
            status="error",
            checkpoint_path=str(checkpoint_path) if 'checkpoint_path' in locals() else "",
            checkpoint_id=checkpoint_id if 'checkpoint_id' in locals() else None,
            evaluation_data_info={},
            results={}
        )

@router.post("/inference")
async def inference_model(
    request: Request,
    payload: schemas.InferenceRequest,
    checkpoint_id: Optional[str] = Query(
        default=None,
        description="Checkpoint identifier (e.g., '10', '10.1'). If not provided, uses latest checkpoint.",
        examples=["1"]
    ),
    threshold_config: Optional[Dict[str, Any]] = Body(
        default=None,
        description="Threshold configuration for predictions",
        examples=[{
            "thresholds": {
                "payment_intent": 0.75,
                "payment_request": 0.70,
                "open_intent": 0.60
            },
            "fallback_label": "Unknown"
        }]
    )
):
    """Run inference on text samples using a trained model with optional threshold-based fallback.

    Args:
        payload: InferenceRequest with text samples
        checkpoint_id: Optional checkpoint identifier (e.g., "10", "10.1"). If not provided, uses latest checkpoint.
        threshold_config: Optional threshold configuration for fallback logic
    """
    trainer_service: services.TrainerService = request.app.state.trainer_service

    # Get checkpoint path - use specific checkpoint if provided, otherwise latest
    if checkpoint_id:
        checkpoint_pth = trainer_service.get_item_path_by_id(checkpoint_id)
        if checkpoint_pth is None:
            return {
                "message": f"Checkpoint {checkpoint_id} not found",
                "status": "error"
            }
    else:
        checkpoint_pth = trainer_service.get_latest_item_path()
        if checkpoint_pth is None:
            return {
                "message": "No trained model checkpoint found",
                "status": "error"
            }

    # Parse threshold config if provided
    parsed_threshold_config = None
    if threshold_config:
        parsed_threshold_config = ThresholdConfig(**threshold_config)
        logger.info(f"Using threshold config: {parsed_threshold_config}")

    # Detect inference type from checkpoint
    inference_type = _detect_inference_type(checkpoint_pth)

    # Use appropriate inferencer
    if inference_type == "prob":
        inferencer = ProbModelInference(
            checkpoint_pth,
            label_config=PAYMENT_LABEL_V2,
            threshold_config=parsed_threshold_config
        )
    else:
        inferencer = ADBModelInference(
            checkpoint_pth,
            label_config=PAYMENT_LABEL_V2,
            threshold_config=parsed_threshold_config
        )

    predictions = inferencer.predict(payload.text)
    output = [
        {"text": text, "predicted_label": label}
        for text, label in zip(payload.text, predictions)
    ]

    model_info = inferencer.info()

    return {
        "message": "Inference completed",
        "status": "completed",
        "results": {
            "checkpoint": str(checkpoint_pth),
            "checkpoint_id": checkpoint_id or Path(checkpoint_pth).name,
            "inference_type": inference_type,
            "predictions": output,
            "model_info": model_info,
            "threshold_config": threshold_config
        }
    }

@router.post("/analyze-error-patterns")
async def analyze_error_patterns(
    request: Request,
    payload: schemas.AnalyzeErrorPatternsRequest = schemas.AnalyzeErrorPatternsRequest()
):
    """
    Analyze error patterns from evaluation results using LLM.
    This endpoint takes the errors_by_label output from model evaluation and
    uses LLM to identify patterns, root causes, and data-focused recommendations.

    Args:
        payload.checkpoint_id: Optional checkpoint identifier (e.g., "1", "2", "1.1", "2.3").
                               If not provided, uses the latest checkpoint.
        payload.iteration_number: Optional evaluation dataset iteration number.
                                  If not provided, uses the latest evaluation dataset.
    """
    # Get services from app state
    model_analyzer: services.ModelAnalyzer = request.app.state.model_analyzer
    trainer_service: services.TrainerService = request.app.state.trainer_service
    eval_data_manager: services.EvalDataManager = request.app.state.eval_data_manager
    error_pattern_analyzer: services.ErrorPatternAnalysisService = request.app.state.error_pattern_analyzer

    try:
        # Get checkpoint path - use specific checkpoint if provided, otherwise latest
        if payload.checkpoint_id:
            checkpoint_path = trainer_service.get_item_path_by_id(payload.checkpoint_id)
            if checkpoint_path is None:
                return {
                    "message": f"Checkpoint {payload.checkpoint_id} not found",
                    "status": "error",
                    "error_analyses": []
                }
            logger.info(f"Using specified checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = trainer_service.get_latest_item_path()
            if checkpoint_path is None:
                return {
                    "message": "No trained model checkpoint found",
                    "status": "error",
                    "error_analyses": []
                }
            logger.info(f"Using latest checkpoint: {checkpoint_path}")

        # Load model and evaluation data
        # For PAYMENT_LABEL_V2, always use probability-based inference
        prob_inferencer = ProbModelInference(peft_path=str(checkpoint_path), label_config=PAYMENT_LABEL_V2)
        model_analyzer.load_model(str(checkpoint_path), inferencer=prob_inferencer)

        # Use specified iteration number or latest
        iteration_number = payload.iteration_number if payload.iteration_number is not None else eval_data_manager.get_latest_item_number()
        evaluation_dataset = eval_data_manager.to_datasets(iteration_number)

        if evaluation_dataset is None:
            return {
                "message": f"No evaluation dataset found for iteration {iteration_number}",
                "status": "error",
                "error_analyses": []
            }

        logger.info(f"Running model evaluation for error pattern analysis using checkpoint: {checkpoint_path}, iteration: {iteration_number}")
        open_intent_samples = eval_data_manager.load_open_intent(iteration_number)

        # Run evaluation to get errors_by_label
        evaluation_result = model_analyzer.analyze_model(
            evaluation_dataset=evaluation_dataset,
            open_intent_samples=open_intent_samples,
            include_test_cases=False,
        )

        if not evaluation_result.errors_by_label:
            return {
                "message": "No errors found to analyze",
                "status": "completed",
                "error_analyses": []
            }

        # Get label explanations from the label config
        label_explanations = PAYMENT_LABEL_V2.get_label_explanation()

        # Analyze error patterns using LLM
        logger.info("Starting LLM-based error pattern analysis...")
        error_analyses = await error_pattern_analyzer.analyze_error_patterns(
            errors_by_label=evaluation_result.errors_by_label,
            label_explanations=label_explanations
        )

        return {
            "message": f"Error pattern analysis completed. Analyzed {len(error_analyses)} error patterns.",
            "status": "completed",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_id": payload.checkpoint_id if payload.checkpoint_id else Path(checkpoint_path).name,
            "iteration_number": iteration_number,
            "error_analyses": [analysis.model_dump() for analysis in error_analyses],
            "evaluation_summary": {
                "overall_accuracy": evaluation_result.overall.accuracy,
                "macro_f1": evaluation_result.overall.macro_f1,
                "total_errors": sum(
                    len(label_errors.false_positives) + len(label_errors.false_negatives)
                    for label_errors in evaluation_result.errors_by_label.values()
                )
            }
        }

    except Exception as e:
        logger.error(f"Error pattern analysis failed: {e}")
        return {
            "message": f"Error pattern analysis failed: {str(e)}",
            "status": "error",
            "error_analyses": []
        }


@router.post("/make-fix-prompt")
async def make_fix_prompt(request: Request, payload: schemas.BuildPromptRequest):
    prompt_builder: services.PromptBuilder = request.app.state.prompt_builder

    prompt = prompt_builder.build_generate_data_prompt(payload.actions)

    if prompt is None:
        return {
            "message": "No prompt generated. Ensure at least one action is of type 'generate_more'.",
            "status": "error",
            "prompt": ""
        }

    return {
        "message": "Prompt generated successfully",
        "status": "completed",
        "prompt": prompt
    }

@router.post("/generate-fix-data")
async def generate_fix_data(request: Request, payload: schemas.FixGenRequest):
    """Generate synthetic data using custom prompt for fixing model issues.

    The generated data is:
    1. Deduplicated and filtered against existing data using ROUGE-L
    2. Appended to the main data file
    3. Saved to a separate timestamped file for tracking
    """
    data_generator_v2: services.DataGeneratorV2 = request.app.state.data_generator_v2
    data_manager: services.DataManager = request.app.state.data_manager

    try:
        # Load human seed data
        path = ".cache/human_seed.json"
        with open(path, "r") as f:
            seed_data = json.load(f)
        logger.info(f"Loaded {len(seed_data)} seed examples from {path}")

        # Convert to Sample objects
        converted = [
            {**seed, 'label': PAYMENT_LABEL_V2.to_str(seed['label'])}
            for seed in seed_data
        ]
        human_seeds = TypeAdapter(List[schemas.Sample]).validate_python(converted)

        # Generate data using the fix_gen method with custom prompt
        # Returns: (all_results, versioned_file_path, saved_count)
        all_results, versioned_file_path, saved_count = await data_generator_v2.fix_gen(
            human_seeds=human_seeds,
            prompt=payload.prompt,
            amount=payload.amount
        )

        return {
            "message": f"Fix data generation completed. Generated {len(all_results)} samples, {saved_count} saved after deduplication.",
            "status": "completed",
            "samples_generated": len(all_results),
            "samples_saved": saved_count,
            "dataset_path": versioned_file_path,  # Path to use with /continual-train
            "main_file": data_manager.LOCAL_FILE,
        }

    except Exception as e:
        logger.error(f"Fix data generation failed: {str(e)}")
        return {
            "message": f"Fix data generation failed: {str(e)}",
            "status": "error",
            "samples_generated": 0,
            "samples_saved": 0
        }

@router.post("/run-orchestrator")
async def run_orchestrator(
    request: Request,
    payload: schemas.OrchestratorRunRequest = schemas.OrchestratorRunRequest()
):
    """
    Run the complete training orchestration pipeline with configurable parameters.

    This endpoint orchestrates an iterative training loop:
    1. Loads model from initial checkpoint (e.g., "11.7")
    2. Evaluates on evaluation dataset (configurable iteration)
    3. Analyzes errors using LLM (error pattern analyzer)
    4. For each data action from error analysis, generates training samples (configurable amount)
    5. Continues training from previous checkpoint (creates sub-versions like 11.8, 11.9, etc.)
    6. Repeats until all labels achieve target F1 (configurable threshold)

    Request Body:
        - initial_checkpoint_id: Starting checkpoint ID (default: "11.7")
        - max_iterations: Maximum iterations to prevent infinite loops (default: 20, range: 1-100)
        - target_f1_per_label: Target F1 score for convergence (default: 0.7, range: 0.0-1.0)
        - samples_per_action: Samples to generate per data action (default: 500, range: 10-2000)
        - iteration_number: Eval dataset iteration to use (default: None = latest)

    Returns:
        Pipeline execution results with final metrics and configuration used

    Example:
        ```json
        {
            "initial_checkpoint_id": "11.7",
            "max_iterations": 20,
            "target_f1_per_label": 0.75,
            "samples_per_action": 600,
            "iteration_number": 1
        }
        ```
    """
    orchestrator: services.TrainingOrchestrator = request.app.state.orchestrator

    try:
        logger.info(f"Starting orchestrator with config: {payload.model_dump()}")

        result = await orchestrator.run(
            initial_checkpoint_id=payload.initial_checkpoint_id,
            max_iterations=payload.max_iterations,
            target_f1_per_label=payload.target_f1_per_label,
            samples_per_action=payload.samples_per_action,
            iteration_number=payload.iteration_number
        )
        return result

    except Exception as e:
        logger.error(f"Orchestrator run failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "status": "error",
            "message": f"Orchestrator run failed: {str(e)}",
            "error": str(e),
            "config": payload.model_dump()
        }