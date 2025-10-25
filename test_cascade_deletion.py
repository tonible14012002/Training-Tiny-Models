#!/usr/bin/env python3
"""
Test script to verify cascade deletion of evaluation results when deleting a trained model.

This script demonstrates that deleting a trained_model automatically deletes
all associated evaluation_result records due to ON DELETE CASCADE.
"""
import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import select

from app.core.repositories import trained_model_repository as tm_repo
from app.core.repositories import evaluation_result_repository as eval_repo
from app.core.models.models import TrainedModel, EvaluationResult
from app.core.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_cascade_deletion():
    """Test that deleting a trained model cascades to evaluation results"""

    # Create database engine
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )

    # Create session maker
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        # Create repositories
        trained_model_repo = tm_repo.TrainedModelRepository(session)
        eval_result_repo = eval_repo.EvaluationResultRepository(session)

        logger.info("=" * 80)
        logger.info("CASCADE DELETION TEST")
        logger.info("=" * 80)
        logger.info("")

        # Find a trained model that has evaluation results
        logger.info("1. Finding a trained model with evaluation results...")

        # Get all trained models
        statement = select(TrainedModel).limit(20)
        result = await session.execute(statement)
        all_models = list(result.scalars().all())

        if not all_models:
            logger.error("No trained models found in database!")
            await engine.dispose()
            return

        # Find one with evaluation results
        test_model = None
        eval_count_before = 0

        for model in all_models:
            eval_results = await eval_result_repo.get_by_trained_model(model.id)
            if eval_results:
                test_model = model
                eval_count_before = len(eval_results)
                logger.info(f"   ✓ Found model: {model.id}")
                logger.info(f"   ✓ Model name: {model.model_name}")
                logger.info(f"   ✓ Evaluation results: {eval_count_before}")
                logger.info("")
                break

        if not test_model:
            logger.warning("No trained models with evaluation results found!")
            logger.info("This is expected if you haven't run evaluations yet.")
            logger.info("")
            logger.info("To test cascade deletion:")
            logger.info("1. Train a model: POST /v2/workflow/train")
            logger.info("2. Evaluate it: POST /v2/workflow/evaluate")
            logger.info("3. Run this script again")
            await engine.dispose()
            return

        # Show the evaluation results
        logger.info("2. Evaluation results before deletion:")
        eval_results_before = await eval_result_repo.get_by_trained_model(test_model.id)
        for i, eval_result in enumerate(eval_results_before):
            logger.info(f"   [{i}] ID: {eval_result.id}")
            logger.info(f"       Accuracy: {eval_result.accuracy}")
            logger.info(f"       F1 Score: {eval_result.f1_score}")
            logger.info(f"       Evaluated at: {eval_result.evaluated_at}")
        logger.info("")

        # Confirm deletion
        logger.info("3. Testing CASCADE deletion...")
        logger.info(f"   This will delete model: {test_model.id}")
        logger.info(f"   Expected: {eval_count_before} evaluation results will also be deleted")
        logger.info("")

        # Actually delete the model (CASCADE will handle evaluation results)
        logger.info("   SIMULATION ONLY - Not actually deleting!")
        logger.info("")
        logger.info("   To actually test, uncomment the deletion code in the script.")
        logger.info("")

        # Uncomment below to actually test deletion:
        # logger.info("   Deleting trained model...")
        # deleted = await trained_model_repo.delete(test_model.id)
        # await session.commit()
        #
        # if deleted:
        #     logger.info("   ✓ Trained model deleted")
        #
        #     # Verify evaluation results are also deleted
        #     eval_results_after = await eval_result_repo.get_by_trained_model(test_model.id)
        #     eval_count_after = len(eval_results_after)
        #
        #     logger.info(f"   ✓ Evaluation results after deletion: {eval_count_after}")
        #     logger.info("")
        #
        #     if eval_count_after == 0:
        #         logger.info("4. ✓ SUCCESS: CASCADE deletion worked!")
        #         logger.info(f"   All {eval_count_before} evaluation results were automatically deleted")
        #     else:
        #         logger.error("4. ✗ FAILED: CASCADE deletion did not work!")
        #         logger.error(f"   Expected 0 evaluation results, found {eval_count_after}")
        # else:
        #     logger.error("   ✗ Failed to delete trained model")

        logger.info("=" * 80)
        logger.info("EXPECTED BEHAVIOR:")
        logger.info("=" * 80)
        logger.info("When you delete a trained_model record, SQLite automatically:")
        logger.info("1. Finds all evaluation_result records with matching trained_model_id")
        logger.info("2. Deletes those evaluation_result records FIRST")
        logger.info("3. Then deletes the trained_model record")
        logger.info("")
        logger.info("This is because of the foreign key constraint:")
        logger.info("  FOREIGN KEY (trained_model_id) REFERENCES trained_model(id) ON DELETE CASCADE")
        logger.info("")
        logger.info("The CASCADE happens at the database level - no application code needed!")
        logger.info("=" * 80)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_cascade_deletion())
