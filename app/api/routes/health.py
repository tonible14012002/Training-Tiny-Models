"""Health check and root endpoints."""

from fastapi import APIRouter
from app.core.settings import settings

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Fine-tuning Workflow API", "version": "1.0.0"}


@router.get("/health-check")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
    }
    