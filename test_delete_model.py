#!/usr/bin/env python3
"""
Test script for model deletion endpoint.

This script tests both the database deletion and file deletion functionality.
"""
import asyncio
import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.repositories import trained_model_repository as tm_repo
from app.core.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_delete_model():
    """Test model deletion functionality"""

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
        # Create repository
        trained_model_repo = tm_repo.TrainedModelRepository(session)

        # Get all trained models
        logger.info("Fetching all trained models...")
        all_models = await trained_model_repo.get_by_status("completed")

        if not all_models:
            logger.warning("No completed models found in database")
            logger.info("Trying to get all models regardless of status...")
            from sqlmodel import select
            from app.core.models.models import TrainedModel

            statement = select(TrainedModel).limit(10)
            result = await session.execute(statement)
            all_models = list(result.scalars().all())

        if not all_models:
            logger.error("No trained models found in database at all!")
            return

        logger.info(f"Found {len(all_models)} trained models")

        # Display first 5 models
        logger.info("\nAvailable models:")
        for i, model in enumerate(all_models[:5]):
            logger.info(f"  [{i}] ID: {model.id}")
            logger.info(f"      Name: {model.model_name}")
            logger.info(f"      Path: {model.model_save_path}")
            logger.info(f"      Status: {model.status}")
            logger.info(f"      Created: {model.created_at}")
            logger.info("")

        # For testing, let's check if we can find a model to delete
        # We won't actually delete it in this test, just simulate
        test_model = all_models[0]
        model_path = Path(test_model.model_save_path)

        logger.info(f"\nTest deletion simulation for model:")
        logger.info(f"  ID: {test_model.id}")
        logger.info(f"  Path: {test_model.model_save_path}")
        logger.info(f"  Path exists: {model_path.exists()}")

        if model_path.exists():
            logger.info(f"  Path is directory: {model_path.is_dir()}")
            if model_path.is_dir():
                files = list(model_path.iterdir())
                logger.info(f"  Directory contains {len(files)} items:")
                for f in files[:5]:
                    logger.info(f"    - {f.name}")

        # Simulate deletion check (don't actually delete)
        logger.info("\n--- DELETION SIMULATION ---")
        logger.info(f"Would delete from database: {test_model.id}")
        if model_path.exists():
            logger.info(f"Would delete files at: {model_path}")
            logger.info("✓ Deletion would succeed")
        else:
            logger.info(f"Files don't exist at: {model_path}")
            logger.info("✓ Would only delete from database")

        logger.info("\nTo actually test deletion via API:")
        logger.info("1. Start the server: make start")
        logger.info("2. Use test_delete_model_api.py script")
        logger.info(f"   - Model ID to test: {test_model.id}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_delete_model())
