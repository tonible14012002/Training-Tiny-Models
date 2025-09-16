import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException

from app.api.dependencies import api_key_auth
from app.core.settings import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Initialize resources before the app starts
    logger.info("Starting up fine-tuning workflow API...")

    # Store settings in app state
    app.state.settings = settings

    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Teacher model: {settings.TEACHER_MODEL}")
    logger.info(f"Student model path: {settings.STUDENT_MODEL_PATH}")

    yield  # The app runs here

    # Cleanup resources when shutting down
    logger.info("Shutting down fine-tuning workflow API...")


# Exception handlers
async def http_exception_handler(request, exc):
    return ORJSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


async def request_validation_exception_handler(request, exc: RequestValidationError):
    return ORJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )


async def unhandled_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return ORJSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Create FastAPI app
app = FastAPI(
    title="Fine-tuning Workflow API",
    description="API for iterative LLM fine-tuning with synthetic data generation",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(api_key_auth)],
    default_response_class=ORJSONResponse
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Fine-tuning Workflow API", "version": "1.0.0"}


# Health check endpoint
@app.get("/health-check")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "teacher_model": settings.TEACHER_MODEL,
        "student_model_path": settings.STUDENT_MODEL_PATH
    }


# Workflow endpoints
@app.post("/workflow/generate-data")
async def generate_synthetic_data():
    """Generate synthetic training data."""
    return {"message": "Data generation started", "status": "in_progress"}


@app.post("/workflow/train")
async def train_student_model():
    """Train the student model with curated data."""
    return {"message": "Training started", "status": "in_progress"}


@app.post("/workflow/evaluate")
async def evaluate_model():
    """Evaluate model performance on dev/test sets."""
    return {"message": "Evaluation started", "status": "in_progress"}


@app.get("/workflow/status")
async def get_workflow_status():
    """Get current workflow status."""
    return {
        "status": "idle",
        "current_loop": 0,
        "total_examples": 0,
        "last_f1_score": None
    }


@app.get("/workflow/metrics")
async def get_metrics():
    """Get training metrics and performance history."""
    return {
        "loops_completed": 0,
        "total_examples_generated": 0,
        "current_f1_score": None,
        "best_f1_score": None,
        "training_history": []
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "dev"
    )