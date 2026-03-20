"""
API route handlers for real-time measurement endpoints
Supports both 2D and 3D object measurement
"""

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
import base64
import logging
from typing import Optional

from app.models.schemas import (
    RealtimeMeasurementResponse,
    RealtimeMeasurementRequest,
    MeasuredObject3DResponse,
    CalibrationRequest,
    CalibrationResponse,
    ErrorResponse,
    ObjectType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["realtime-measurement"])

# Lazy import to avoid loading models at startup
_processor = None


def get_processor():
    """Get or create processor instance"""
    global _processor
    if _processor is None:
        from app.utils.realtime_processor import RealtimeProcessor

        _processor = RealtimeProcessor(
            use_gpu=False, model_type="midas_small", confidence_threshold=0.4
        )
    return _processor


@router.post(
    "/measure",
    response_model=RealtimeMeasurementResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Real-time object measurement (2D/3D)",
    description="""
    Measure objects in real-time from camera frames.
    
    **Features:**
    - Automatic 2D vs 3D object detection
    - 2D objects: Returns Length and Breadth
    - 3D objects: Returns Length, Breadth, and Height
    - Uses AI depth estimation (no special sensors needed)
    - Works with any camera
    
    **For best results:**
    - Keep camera steady
    - Good lighting conditions
    - Objects should be clearly visible
    - Provide calibration_distance_cm if known
    
    **Returns:**
    - List of measured objects with dimensions in centimeters
    - Annotated image with measurements drawn
    """,
)
async def measure_realtime(request: RealtimeMeasurementRequest):
    """Process frame and measure all visible objects"""

    try:
        # Decode base64 image
        image_data = request.image

        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid base64 encoding: {str(e)}"
            )

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please ensure it's a valid image file.",
            )

        # Get processor and process frame
        processor = get_processor()

        # Apply calibration if provided
        if request.calibration_distance_cm:
            processor.calibrate(reference_distance_cm=request.calibration_distance_cm)

        # Process the frame
        result = processor.process_frame(
            image, return_annotated=request.return_annotated
        )

        # Convert to response format
        objects_response = [
            MeasuredObject3DResponse(
                object_id=obj.object_id,
                object_type=ObjectType(obj.object_type.value),
                label=obj.label,
                confidence=round(obj.confidence, 2),
                length_cm=obj.length_cm,
                breadth_cm=obj.breadth_cm,
                height_cm=obj.height_cm,
                bounding_box=obj.bounding_box,
                center=obj.center,
                depth_value=round(obj.depth_value, 3),
            )
            for obj in result.objects
        ]

        return RealtimeMeasurementResponse(
            success=result.success,
            message=result.message,
            objects=objects_response,
            frame_width=result.frame_width,
            frame_height=result.frame_height,
            processing_time_ms=result.processing_time_ms,
            annotated_image=result.annotated_image_base64,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in realtime measurement: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post(
    "/calibrate",
    response_model=CalibrationResponse,
    summary="Calibrate measurement system",
    description="""
    Calibrate the measurement system with known reference values.
    
    Provide the distance from camera to objects and optionally
    a reference object with known dimensions for more accurate measurements.
    """,
)
async def calibrate_measurement(request: CalibrationRequest):
    """Calibrate the measurement system"""

    try:
        processor = get_processor()

        processor.calibrate(reference_distance_cm=request.reference_distance_cm)

        # If reference object dimensions provided, calculate scale factor
        scale_factor = processor._depth_scale_factor

        if request.reference_object_width_cm and request.reference_object_height_cm:
            # Store for next calibration with image
            processor._pending_calibration = (
                request.reference_object_width_cm,
                request.reference_object_height_cm,
            )

        return CalibrationResponse(
            success=True,
            message=f"Calibration set: distance={request.reference_distance_cm}cm",
            scale_factor=scale_factor,
        )

    except Exception as e:
        logger.error(f"Calibration error: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@router.get(
    "/status",
    summary="Get processor status",
    description="Check if models are loaded and ready",
)
async def get_status():
    """Get current processor status"""

    try:
        processor = get_processor()

        models_loaded = (
            processor._depth_model is not None
            and processor._object_detector is not None
        )

        return {
            "ready": True,
            "models_loaded": models_loaded,
            "device": str(processor.device),
            "depth_model": processor.model_type,
            "confidence_threshold": processor.confidence_threshold,
            "calibration": {
                "reference_distance_cm": processor.reference_distance_cm,
                "scale_factor": processor._depth_scale_factor,
            },
        }

    except Exception as e:
        return {"ready": False, "error": str(e)}


@router.post(
    "/warmup",
    summary="Warm up models",
    description="Pre-load models for faster first inference",
)
async def warmup_models():
    """Pre-load ML models"""

    try:
        processor = get_processor()

        # Create a dummy image to trigger model loading
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # This will load both models
        processor.process_frame(dummy_image, return_annotated=False)

        return {"success": True, "message": "Models loaded and ready"}

    except Exception as e:
        logger.error(f"Warmup error: {e}")
        return {"success": False, "message": f"Warmup failed: {str(e)}"}
