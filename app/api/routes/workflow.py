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
    """Generate fresh evaluation data using data generator."""
    data_generator: services.DataGenerator = request.app.state.data_generator
    path = ".cache/human_seed.json"

    # Load human seed data (same as training data generation)
    with open(path, "r") as f:
        seed_data = json.load(f)
    logger.info(f"Loaded {len(seed_data)} seed examples from {path}")

    converted = [
        {**seed, 'label': schemas.PAYMENT_LABEL.to_str(seed['label'])}
        for seed in seed_data
    ]

    human_seeds = TypeAdapter(List[schemas.Sample]).validate_python(converted)

    # Generate fresh evaluation data
    eval_samples = await data_generator.fresh_gen_eval(human_seeds)

    return {
        "message": "Evaluation data generation completed",
        "status": "completed",
        "samples_generated": len(eval_samples)
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
async def evaluate_model(request: Request):
    """Evaluate model performance on dev/test sets."""
    data_generator: services.DataGenerator = request.app.state.data_generator
    model_analyzer: services.ModelAnalyzer = request.app.state.model_analyzer
    trainer_service: services.TrainerService = request.app.state.trainer_service

    trainer_service.load_model()

    return {
        "message": "Evaluation completed",
        "status": "completed",
    }

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