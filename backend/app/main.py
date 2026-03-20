"""
Main FastAPI application entry point
Real-time Object Measurement API with 2D/3D support
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv

from app.routes import measurement, realtime_measurement
from app.models.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API metadata
API_TITLE = "Object Measurement API"
API_DESCRIPTION = """
## Real-time Object Measurement API

Measure real-world object dimensions using AI-powered depth estimation.

### Features:
- **Real-time measurement** - Process camera frames instantly
- **2D/3D detection** - Automatically detects flat vs volumetric objects
- **No reference needed** - Works without A4 paper or markers
- **AI depth estimation** - Uses MiDaS neural network for depth perception

### Measurement Types:

**2D Objects (flat):**
- Length (cm)
- Breadth (cm)

**3D Objects (volumetric):**
- Length (cm)
- Breadth (cm)
- Height (cm)

### API Versions:

- **v1** - Legacy API with A4 paper reference (backward compatible)
- **v2** - New real-time API with AI depth estimation

### For best results:
- Good lighting conditions
- Keep camera steady
- Objects should be clearly visible
- Calibrate with known distance if possible
"""
API_VERSION = "2.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info("Real-time 2D/3D measurement enabled")

    # Pre-warm models in background (optional)
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        logger.info("Pre-loading ML models...")
        try:
            from app.utils.realtime_processor import get_processor

            processor = get_processor()
            import numpy as np

            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            processor.process_frame(dummy, return_annotated=False)
            logger.info("Models pre-loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to pre-load models: {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS - allow all origins for mobile app
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# For development/mobile apps, we need permissive CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for mobile app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(measurement.router)  # Legacy v1 API
app.include_router(realtime_measurement.router)  # New v2 API


@app.get("/", tags=["health"])
async def root():
    """Root endpoint - API info"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "v1": "/api/v1/measure (legacy - A4 reference)",
            "v2": "/api/v2/measure (realtime - AI depth)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint for monitoring"""

    # Check if models are loaded
    models_loaded = False
    try:
        from app.utils.realtime_processor import _processor

        if _processor is not None:
            models_loaded = (
                _processor._depth_model is not None
                and _processor._object_detector is not None
            )
    except:
        pass

    return HealthResponse(
        status="healthy", version=API_VERSION, models_loaded=models_loaded
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    uvicorn.run("app.main:app", host=host, port=port, reload=debug)
