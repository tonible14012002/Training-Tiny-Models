from fastapi import APIRouter, Request, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core import repositories as repos
from app.api.dependencies import get_db_session
from app.core import services
from app.core import schemas
from src.payment_classifier.inference.prob_inference import ProbModelInference
from app.utils.dataset_helper import DatasetHelper

import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflow", tags=["workflow"])

@router.post("/train")
async def train_model(
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    data_manager: services.DataManager = request.app.state.data_manager
    ds = data_manager.to_datasets()

    checkpoint_path = await services.TrainerService(
        base_model="prajjwal1/bert-tiny",
        label_config=request.app.state.label_config,
    ).train(ds)

    return {"checkpoint_path": checkpoint_path}

@router.post("/convert/onnx")
async def convert_to_onnx(
    pipeline_id: str = Body(..., description="The ID of the pipeline to convert"),
    path: str = Body(..., description="The path to save the ONNX model"),
    db: AsyncSession = Depends(get_db_session),
    request: Request = None,
):
    # Merge the models
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    model = ORTModelForSequenceClassification.from_pretrained(
        path, 
        export=True,
    )
    model.save_pretrained(path)

    return {
        "message": f"Model converted and saved to {path}",
        "data":{
            "saved_path": path + "/onnx/model.onnx"
        }
    }

@router.post("first-gen")
async def first_generation(
    request: Request = None,
):
    data_generator_v2: services.DataGeneratorV2 = request.app.state.data_generator_v2
    label_config = request.app.state.label_config

    # Load human seed
    with open("./.cache/human_seed.json", "r") as f:
        human_seed = json.loads(f.read()) 
    
    human_samples = [
        schemas.Sample(
            msg=msg["msg"],
            label= label_config.to_str(msg["label"]),
        ) for msg in human_seed
    ]

    # Create dict
    label_dict = label_config.to_dict()
    for key in label_dict:
        label_dict[key] = 200

    generated = await data_generator_v2.fresh_gen_v2(
        human_seeds=human_samples,
        expect_total_each_label=label_dict
    )

    return {
        "message": f"Generated {len(generated)} samples",
    }

@router.post("/evaluate-with-frozen-set")
async def evaluate_human_set(
    request: Request = None,
    payload: dict = Body(..., description="The payload containing the frozen test set")
):
    label_config = request.app.state.label_config
    with open("./.cache/frozen_test_set.json", "r") as f:
        frozen_set = json.loads(f.read())

    eval_ds = DatasetHelper.json_to_ds(
        json_list=frozen_set,
        label_config=label_config
    )

    inferencer = ProbModelInference(
        peft_path="/Users/maroon/workspace/tiny-model-tunning/finetune/.checkpoints/payment_classification_v2/" + payload["checkpoint_id"],
        label_config=label_config,
    )

    result = inferencer.evaluate(
        eval_ds,
        get_low_confidence=True,
        get_error_samples=True,
        confidence_threshold=0.5
    )

    return {
        "message": "Evaluation completed",
        "data": result
    }


@router.post("/classify-error")
async def  classify_errors(
    request: Request = None,
):
    pass