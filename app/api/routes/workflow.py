"""Workflow endpoints."""

from fastapi import APIRouter, Request
from app.core import services
from app.core import schemas
from app.core.schemas.workflow import PAYMENT_LABEL_V2
from src.payment_classifier.inference import ADBModelInference
from src.payment_classifier.inference.prob_inference import ProbModelInference

import json
import logging
from pydantic import TypeAdapter
from typing import List
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
            return config.get("inference_type", "adb")
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Could not read inference config from {config_file}, defaulting to ADB")

    # Fallback: check if ADB data exists
    adb_file = Path(checkpoint_path) / "adb_data.json"
    if adb_file.exists():
        return "adb"

    # Default to ADB for backward compatibility
    return "adb"

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
async def train_student_model(request: Request, inference_type: str = "adb"):
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
        # For PAYMENT_LABEL_V2, always use probability-based inference
        prob_inferencer = ProbModelInference(peft_path=str(checkpoint_path), label_config=PAYMENT_LABEL_V2)
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

    if checkpoint_pth is None:
        return {
            "message": "No trained model checkpoint found",
            "status": "error"
        }

    # Detect inference type from checkpoint
    inference_type = _detect_inference_type(checkpoint_pth)

    # Use appropriate inferencer
    if inference_type == "prob":
        inferencer = ProbModelInference(checkpoint_pth, label_config=PAYMENT_LABEL_V2)
    else:
        inferencer = ADBModelInference(checkpoint_pth, label_config=PAYMENT_LABEL_V2)

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
            "checkpoint": checkpoint_pth,
            "inference_type": inference_type,
            "predictions": output,
            "model_info": model_info
        }
    }

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
        # For PAYMENT_LABEL_V2, always use probability-based inference
        prob_inferencer = ProbModelInference(peft_path=str(checkpoint_path), label_config=PAYMENT_LABEL_V2)
        model_analyzer.load_model(str(checkpoint_path), inferencer=prob_inferencer)
        evaluation_dataset = eval_data_manager.to_datasets()  # Load latest evaluation data

        if evaluation_dataset is None:
            return {
                "message": "No evaluation dataset found",
                "status": "error",
                "error_analyses": []
            }

        logger.info(f"Running model evaluation for error pattern analysis using checkpoint: {checkpoint_path}")
        iteration_number = eval_data_manager.get_latest_item_number()
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
            "checkpoint_path": checkpoint_path,
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
    """Generate synthetic data using custom prompt for fixing model issues."""
    data_generator_v2: services.DataGeneratorV2 = request.app.state.data_generator_v2

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
        results = await data_generator_v2.fix_gen(
            human_seeds=human_seeds,
            prompt=payload.prompt,
            amount=payload.amount
        )

        return {
            "message": f"Fix data generation completed. Generated {len(results)} samples.",
            "status": "completed",
            "samples_generated": len(results)
        }

    except Exception as e:
        logger.error(f"Fix data generation failed: {str(e)}")
        return {
            "message": f"Fix data generation failed: {str(e)}",
            "status": "error",
            "samples_generated": 0
        }