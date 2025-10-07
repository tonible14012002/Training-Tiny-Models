import logging
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException

from app.api.dependencies import api_key_auth
from app.api.routes import health, workflow
from app.api.routes import v2 as routes_v2
from app.core.settings import settings, RELOAD_DIRS
from app.core import services
from app.core.schemas.workflow import PAYMENT_LABEL_V2

from src.payment_classifier.llm.litellm import LiteLLMProvider
from src.payment_classifier.llm.settings import LLMSettings
from src.payment_classifier.prompts import InmemoryPromptManager

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Initialize resources before the app starts
    logger.info("Starting up fine-tuning workflow API...")
    label_config = PAYMENT_LABEL_V2

    # Store settings in app state
    app.state.settings = settings

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        connect_args={"statement_cache_size": 0},
    )

    teacher_llm = LiteLLMProvider(LLMSettings(
        llm_model_name="gpt-4o-mini",
        api_key=settings.OPENAI_API_KEY,
        temperature=0.7,
        num_retries=2,
    ))


    prompt_mgr = InmemoryPromptManager()

    logger.info("Initializing data manager...")
    data_manager = services.DataManager(label_config)

    logger.info("Initializing data generator v2...")
    data_generator_v2 = services.DataGeneratorV2(
        llm=teacher_llm,
        prompt_mgr=prompt_mgr,
        data_manager=data_manager,
    )

    trainer_service =  services.TrainerService(
        base_model="prajjwal1/bert-tiny",
        label_config=label_config
    )

    logger.info("Initializing training orchestrator...")

    # Store database engine for session creation
    app.state.engine = engine

    app.state.data_manager = data_manager
    app.state.data_generator_v2 = data_generator_v2
    app.state.trainer_service = trainer_service
    app.state.prompt_mgr = prompt_mgr
    app.state.label_config = label_config

    logger.info("Application startup complete.")

    yield  # The app runs here

    # Cleanup resources when shutting down
    logger.info("Shutting down fine-tuning workflow API...")
    await engine.dispose()


# Exception handlers
async def http_exception_handler(_, exc):
    return ORJSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


async def request_validation_exception_handler(_, exc: RequestValidationError):
    return ORJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )


async def unhandled_exception_handler(_, exc):
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

# Include routers
app.include_router(health.router)
app.include_router(workflow.router)
app.include_router(routes_v2.workflow.router, prefix="/v2", tags=["v2"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "dev",
        reload_dirs=RELOAD_DIRS,
    )