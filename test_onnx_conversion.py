#!/usr/bin/env python3
"""
Test script for ONNX conversion endpoint.

This script tests the ONNX conversion functionality without requiring the server to be running.
"""
import asyncio
import logging
from pathlib import Path
import shutil
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_onnx_conversion():
    """Test ONNX conversion with a real model"""

    # Use an existing model path
    model_path = Path(".checkpoints/payment_classification_v2/16")

    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        logger.info("Available models:")
        checkpoints_dir = Path(".checkpoints/payment_classification_v2")
        if checkpoints_dir.exists():
            for item in sorted(checkpoints_dir.iterdir()):
                if item.is_dir():
                    logger.info(f"  - {item}")
        return

    merged_path = model_path / "_merged"
    if not merged_path.exists():
        logger.error(f"Merged model not found: {merged_path}")
        return

    logger.info(f"Testing ONNX conversion for: {model_path}")
    logger.info(f"Merged model path: {merged_path}")

    # Create output directory with new structure
    output_name = "test_onnx_model"
    temp_output_dir = Path(".cache/onnx") / f"{output_name}_temp"

    # Clean up existing directory
    if temp_output_dir.exists():
        logger.info(f"Removing existing temp directory: {temp_output_dir}")
        shutil.rmtree(temp_output_dir)

    # Create directory structure
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    onnx_subdir = temp_output_dir / "onnx"
    onnx_subdir.mkdir(parents=True, exist_ok=True)

    try:
        # Load and convert model
        logger.info("Loading model for ONNX conversion...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            str(merged_path),
            export=True,
        )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Save tokenizer to root directory
        logger.info(f"Saving tokenizer to {temp_output_dir}")
        tokenizer.save_pretrained(str(temp_output_dir))

        # Save ONNX model to onnx/ subdirectory
        logger.info(f"Saving ONNX model to {onnx_subdir}")
        ort_model.save_pretrained(str(onnx_subdir))

        # Verify saved files
        logger.info("Verifying saved files in root:")
        for item in sorted(temp_output_dir.iterdir()):
            if item.is_file():
                logger.info(f"  - {item.name}")
            else:
                logger.info(f"  - {item.name}/ (directory)")

        logger.info("Verifying ONNX files in onnx/ subdirectory:")
        for item in sorted(onnx_subdir.iterdir()):
            logger.info(f"  - onnx/{item.name}")

        # Create zip file
        zip_path = Path(".cache/onnx") / f"{output_name}.zip"
        logger.info(f"Creating zip file: {zip_path}")

        if zip_path.exists():
            zip_path.unlink()

        shutil.make_archive(
            str(zip_path.with_suffix('')),
            'zip',
            str(temp_output_dir)
        )

        logger.info(f"✓ ONNX conversion successful!")
        logger.info(f"✓ Output directory: {temp_output_dir}")
        logger.info(f"✓ Zip file: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")

        # Verify zip structure
        logger.info("\nVerifying zip file structure:")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for name in sorted(zip_ref.namelist()):
                logger.info(f"  {name}")

        # Test loading the ONNX model
        logger.info("\nTesting ONNX model loading from subdirectory...")
        test_model = ORTModelForSequenceClassification.from_pretrained(str(onnx_subdir))
        test_tokenizer = AutoTokenizer.from_pretrained(str(temp_output_dir))

        # Test inference
        test_text = "pay john $50"
        inputs = test_tokenizer(test_text, return_tensors="pt")
        outputs = test_model(**inputs)

        logger.info(f"✓ ONNX model loaded and tested successfully!")
        logger.info(f"  Test text: '{test_text}'")
        logger.info(f"  Output shape: {outputs.logits.shape}")

    except Exception as e:
        logger.error(f"Error during ONNX conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_onnx_conversion())
