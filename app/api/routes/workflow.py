"""Workflow endpoints."""

from fastapi import APIRouter, Request
from app.core import services
from app.core import schemas
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
async def evaluate_model():
    """Evaluate model performance on dev/test sets."""
    return {"message": "Evaluation started", "status": "in_progress"}


@router.get("/status")
async def get_workflow_status():
    """Get current workflow status."""
    return {
        "status": "idle",
        "current_loop": 0,
        "total_examples": 0,
        "last_f1_score": None
    }


@router.get("/metrics")
async def get_metrics():
    """Get training metrics and performance history."""
    return {
        "loops_completed": 0,
        "total_examples_generated": 0,
        "current_f1_score": None,
        "best_f1_score": None,
        "training_history": []
    }