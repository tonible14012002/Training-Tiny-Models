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
    checkpoint_path = trainer_service._file_helper.get_item_path_by_id(checkpoint_id)
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
        checkpoint_pth = trainer_service._file_helper.get_item_path_by_id(checkpoint_id)
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