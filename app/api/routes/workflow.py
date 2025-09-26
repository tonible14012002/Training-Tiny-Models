"""Workflow endpoints."""

from fastapi import APIRouter, Request
from app.core import services
from app.core import schemas
from src.payment_classifier.inference import ADBModelInference

import json
import logging
from pydantic import TypeAdapter
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflow", tags=["workflow"])

@router.post("/generate-data")
async def generate_synthetic_data(request: Request):
    data_generator: services.DataGenerator = request.app.state.data_generator
    path = ".cache/human_seed.json"
    # load text from json file
    with open(path, "r") as f:
        seed_data = json.load(f)
    logger.info(f"Loaded {len(seed_data)} seed examples from {path}")
    converted = [
        {**seed, 'label': schemas.PAYMENT_LABEL.to_str(seed['label'])}
        for seed in seed_data
    ]

    human_seeds = TypeAdapter(List[schemas.Sample]).validate_python(converted)
    await data_generator.fresh_gen(human_seeds)

    return {"message": "Data generation started", "status": "in_progress"}

@router.post("/fresh-generate-eval")
async def generate_fresh_eval_data(request: Request):
    """Generate fresh evaluation data using eval generator - both intent and open intent."""
    eval_generator: services.EvalGenerator = request.app.state.eval_generator
    path = ".cache/human_seed.json"

    # Load human seed data for intent generation
    with open(path, "r") as f:
        seed_data = json.load(f)
    logger.info(f"Loaded {len(seed_data)} seed examples from {path}")

    converted = [
        {**seed, 'label': schemas.PAYMENT_LABEL.to_str(seed['label'])}
        for seed in seed_data
    ]

    human_seeds = TypeAdapter(List[schemas.Sample]).validate_python(converted)

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
async def train_student_model(request: Request):
    data_manager: services.DataManager = request.app.state.data_manager
    trainer_service: services.TrainerService = request.app.state.trainer_service

    ds = data_manager.to_datasets()
    await trainer_service.train(ds)

    return {
        "status": "training"
    }

@router.post("/evaluate")
async def evaluate_model(
    request: Request,
    payload: schemas.EvaluationRequest = schemas.EvaluationRequest()
) -> schemas.EvaluationResponse:
    """Evaluate model performance using the latest evaluation dataset with comprehensive analysis."""

    # Get services from app state
    model_analyzer: services.ModelAnalyzer = request.app.state.model_analyzer
    trainer_service: services.TrainerService = request.app.state.trainer_service
    eval_data_manager: services.EvalDataManager = request.app.state.eval_data_manager

    try:
        # Get latest checkpoint
        checkpoint_path = trainer_service.get_latest_item_path()
        if checkpoint_path is None:
            return schemas.EvaluationResponse(
                message="No trained model checkpoint found",
                status="error",
                checkpoint_path="",
                evaluation_data_info={},
                results={}
            )

        logger.info(f"Using checkpoint: {checkpoint_path}")

        # Load evaluation dataset
        eval_samples = eval_data_manager.load(payload.iteration_number)
        if not eval_samples:
            return schemas.EvaluationResponse(
                message="No evaluation data found",
                status="error",
                checkpoint_path=str(checkpoint_path),
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

        # Load model and run comprehensive evaluation
        model_analyzer.load_model(str(checkpoint_path))

        evaluation_result = model_analyzer.analyze_model(
            evaluation_dataset=eval_dataset,
            open_intent_samples=open_intent_samples,
            include_test_cases=payload.include_test_cases
        )

        # # Generate analysis report
        # analysis_report = model_analyzer.generate_analysis_report(evaluation_result)

        # Get error patterns grouped by (expected, predicted) tuples
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
            # "errors_by_label": {k: v.model_dump() for k, v in evaluation_result.errors_by_label.items()} if evaluation_result.errors_by_label else None,
            "error_patterns": {k: [case.model_dump() for case in cases] for k, cases in error_patterns.items()},
            # "analysis_report": analysis_report,
            # "error_buckets": [bucket.model_dump() for bucket in error_buckets],
            # "total_test_cases": len(evaluation_result.test_cases) if evaluation_result.test_cases else 0,
            # "test_cases": [tc.model_dump() for tc in evaluation_result.test_cases] if evaluation_result.test_cases else []
        }

        return schemas.EvaluationResponse(
            message="Model evaluation completed successfully",
            status="completed",
            checkpoint_path=str(checkpoint_path),
            evaluation_data_info=eval_info,
            results=results
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return schemas.EvaluationResponse(
            message=f"Evaluation failed: {str(e)}",
            status="error",
            checkpoint_path=str(checkpoint_path) if 'checkpoint_path' in locals() else "",
            evaluation_data_info={},
            results={}
        )

@router.post("/inference")
async def inference_model(request: Request, payload: schemas.InferenceRequest):
    # Get latest checkpoint
    trainer_service: services.TrainerService = request.app.state.trainer_service
    checkpoint_pth = trainer_service.get_latest_item_path()
    data_manager: services.DataManager = request.app.state.data_manager

    # Run inference
    inferencer = ADBModelInference(checkpoint_pth)
    predictions = inferencer.predict_with_adb(payload.text)
    output = [
        {"text": text, "predicted_label": label}
        for text, label in zip(payload.text, predictions)
    ]

    adb_info = inferencer.info()

    return {
        "message": "Inference completed",
        "status": "completed",
        "results": {
            "checkpiont": checkpoint_pth,
            "predictions": output,
            "adb_info": adb_info
        }
    }

@router.post("/calc-adb")
async def calc_adb(request: Request):
    """Calculate ADB metric on dev/test sets."""
    trainer_service: services.TrainerService = request.app.state.trainer_service
    data_manager: services.DataManager = request.app.state.data_manager

    ds = data_manager.to_datasets()

    checkpoint_pth = trainer_service.get_latest_item_path()
    print(f"Using checkpoint: {checkpoint_pth}")

    inferencer = ADBModelInference(checkpoint_pth)
    inferencer.calc_adb(ds)

    return {
        "message": "ADB calculation completed",
        "status": "completed",
    }

@router.post("/evaluate-adb")
async def evaluate_adb(request: Request):
    """Evaluate model performance using ADB on current dataset."""
    trainer_service: services.TrainerService = request.app.state.trainer_service
    data_manager: services.DataManager = request.app.state.data_manager

    # Get dataset and latest checkpoint
    ds = data_manager.to_datasets()
    checkpoint_pth = trainer_service.get_latest_item_path()
    logger.info(f"Using checkpoint: {checkpoint_pth}")

    # Run ADB evaluation
    inferencer = ADBModelInference(checkpoint_pth)
    evaluation_results = inferencer.evaluate_with_adb(ds)

    return {
        "message": "ADB evaluation completed",
        "status": "completed",
        "results": evaluation_results
    }

@router.post("/auto-train-pipeline")
async def start_auto_training_pipeline(
    request: Request,
    max_iterations: int = 10,
    target_accuracy: float = 0.85,
    target_macro_f1: float = 0.80,
    early_termination_threshold: float = 0.02,
    data_generation_batch_size: int = 15
):
    """
    Start the automated training pipeline that iteratively:
    1. Generates new training data
    2. Trains model with updated dataset
    3. Evaluates model performance
    4. Checks for improvement and termination conditions
    5. Repeats until target metrics are achieved or max iterations reached

    Pipeline includes early termination if no significant improvement is seen
    over multiple consecutive iterations.
    """
    training_orchestrator: services.TrainingOrchestrator = request.app.state.training_orchestrator

    # Create pipeline configuration
    from app.core.schemas.orchestrator import PipelineConfig

    config = PipelineConfig(
        max_iterations=max_iterations,
        target_accuracy=target_accuracy,
        target_macro_f1=target_macro_f1,
        early_termination_threshold=early_termination_threshold,
        data_generation_batch_size=data_generation_batch_size
    )

    # Start the pipeline
    result = await training_orchestrator.start_auto_training_pipeline(config)

    return result


@router.post("/analyze-error-patterns")
async def analyze_error_patterns(request: Request):
    """
    Analyze error patterns from the latest evaluation results using LLM.
    This endpoint takes the errors_by_label output from model evaluation and
    uses LLM to identify patterns, root causes, and data-focused recommendations.
    """
    # Get services from app state
    model_analyzer: services.ModelAnalyzer = request.app.state.model_analyzer
    trainer_service: services.TrainerService = request.app.state.trainer_service
    eval_data_manager: services.EvalDataManager = request.app.state.eval_data_manager
    error_pattern_analyzer: services.ErrorPatternAnalysisService = request.app.state.error_pattern_analyzer

    try:
        # Get latest checkpoint
        checkpoint_path = trainer_service.get_latest_item_path()
        if checkpoint_path is None:
            return {
                "message": "No trained model checkpoint found",
                "status": "error",
                "error_analyses": []
            }

        # Load model and get latest evaluation data
        model_analyzer.load_model(checkpoint_path)
        evaluation_dataset = eval_data_manager.to_datasets()  # Load latest evaluation data

        if evaluation_dataset is None:
            return {
                "message": "No evaluation dataset found",
                "status": "error",
                "error_analyses": []
            }

        logger.info(f"Running model evaluation for error pattern analysis using checkpoint: {checkpoint_path}")

        # Run evaluation to get errors_by_label
        evaluation_result = model_analyzer.analyze_model(
            evaluation_dataset=evaluation_dataset,
            include_test_cases=False,
            use_comprehensive_unknown_evaluation=False
        )

        if not evaluation_result.errors_by_label:
            return {
                "message": "No errors found to analyze",
                "status": "completed",
                "error_analyses": []
            }

        # Define label explanations for payment classification
        label_explanations = {
            "PAYMENT_REQUEST": "User asking someone to send them money",
            "PAYMENT_SEND": "User intends to send/pay money to someone",
            "PAYMENT_COMMAND": "User instructing a system to make a payment",
            "NO_PAYMENT": "No payment intention present"
        }

        # Analyze error patterns using LLM
        logger.info("Starting LLM-based error pattern analysis...")
        error_analyses = await error_pattern_analyzer.analyze_error_patterns(
            errors_by_label=evaluation_result.errors_by_label,
            label_explanations=label_explanations
        )

        return {
            "message": f"Error pattern analysis completed. Analyzed {len(error_analyses)} error patterns.",
            "status": "completed",
            "checkpoint_path": checkpoint_path,
            "error_analyses": [analysis.dict() for analysis in error_analyses],
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

