"""
Main FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.routes import measurement
from app.models.schemas import HealthResponse

# Load environment variables
load_dotenv()

# API metadata
API_TITLE = "Object Measurement API"
API_DESCRIPTION = """
## Object Size Measurement API

Measure real-world object dimensions using computer vision and an A4 paper reference.

### How it works:
1. Place objects on an A4 paper (standard 210mm × 297mm)
2. Take a photo ensuring the entire paper is visible
3. Upload the image to this API
4. Get precise measurements in centimeters

### Features:
- Automatic A4 paper detection
- Perspective correction for angled shots
- Multiple object measurement in single image
- Annotated result image with measurements
"""
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    print(f"🚀 {API_TITLE} v{API_VERSION} starting...")
    yield
    # Shutdown
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,capacitor://localhost,http://localhost",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(measurement.router)


@app.get("/", tags=["health"])
async def root():
    """Root endpoint - redirects to docs"""
    return {"message": "Object Measurement API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(status="healthy", version=API_VERSION)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    uvicorn.run("app.main:app", host=host, port=port, reload=debug)
