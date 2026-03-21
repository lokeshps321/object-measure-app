"""
API route handlers for real-time measurement endpoints
Supports both 2D and 3D object measurement with reference calibration
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
    CalibrationInfo,
    ErrorResponse,
    ObjectType,
    ReferenceType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["realtime-measurement"])

# Lazy import to avoid loading at startup
_processor = None


def get_processor():
    """Get or create processor instance"""
    global _processor
    if _processor is None:
        from app.utils.realtime_processor import RealtimeProcessor

        _processor = RealtimeProcessor(confidence_threshold=0.35)
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
    Measure objects in real-time from camera frames with automatic calibration.
    
    **For accurate measurements:**
    1. Place a credit card (or A4 paper) next to objects
    2. The system auto-detects the reference and calibrates
    3. All objects in frame are measured accurately
    
    **Without reference:**
    - Measurements are estimates based on camera distance
    - Add calibration_distance_cm for better estimates
    
    **Features:**
    - Automatic 2D vs 3D object detection
    - Reference object auto-detection (credit card/A4)
    - Real-time calibration for accurate measurements
    
    **Returns:**
    - List of measured objects with dimensions in centimeters
    - Calibration info showing accuracy status
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

        # Apply manual calibration if distance provided
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
                width_cm=obj.width_cm,
                height_cm=obj.height_cm,
                bounding_box=obj.bounding_box,
                center=obj.center,
            )
            for obj in result.objects
        ]

        # Build calibration info
        calibration_info = None
        if result.calibration_info:
            calibration_info = CalibrationInfo(
                reference_detected=result.calibration_info.get(
                    "reference_detected", False
                ),
                reference_type=result.calibration_info.get("reference_type", "none"),
                pixels_per_cm=result.calibration_info.get("pixels_per_cm", 0),
                reference_width_cm=result.calibration_info.get("reference_width_cm"),
                reference_height_cm=result.calibration_info.get("reference_height_cm"),
            )

        return RealtimeMeasurementResponse(
            success=result.success,
            message=result.message,
            objects=objects_response,
            frame_width=result.frame_width,
            frame_height=result.frame_height,
            processing_time_ms=result.processing_time_ms,
            annotated_image=result.annotated_image_base64,
            calibration_info=calibration_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in realtime measurement: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post(
    "/calibrate",
    response_model=CalibrationResponse,
    summary="Configure measurement calibration",
    description="""
    Configure the measurement system calibration settings.
    
    **Reference Types:**
    - `credit_card`: Standard credit card (8.56 x 5.398 cm) - Most common
    - `a4_paper`: A4 paper sheet (21.0 x 29.7 cm) - For larger objects
    - `custom`: Specify your own reference dimensions
    
    **Usage:**
    1. Call this endpoint to set reference type
    2. Place reference object in frame with objects to measure
    3. System auto-detects reference and calibrates
    """,
)
async def calibrate_measurement(request: CalibrationRequest):
    """Configure the measurement calibration"""

    try:
        processor = get_processor()

        # Convert reference type
        ref_type = (
            request.reference_type.value if request.reference_type else "credit_card"
        )

        # Set calibration
        processor.calibrate(
            reference_distance_cm=request.reference_distance_cm,
            scale_factor=1.0,
            reference_type=ref_type,
            reference_width_cm=request.reference_object_width_cm,
            reference_height_cm=request.reference_object_height_cm,
        )

        message = f"Calibration configured: reference={ref_type}"
        if request.reference_type == ReferenceType.CUSTOM:
            message += f" ({request.reference_object_width_cm}x{request.reference_object_height_cm}cm)"

        return CalibrationResponse(
            success=True,
            message=message,
            scale_factor=1.0,
        )

    except Exception as e:
        logger.error(f"Calibration error: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@router.get(
    "/status",
    summary="Get processor status",
    description="Check processor status and current calibration settings",
)
async def get_status():
    """Get current processor status"""

    try:
        processor = get_processor()

        return {
            "ready": True,
            "models_loaded": processor._models_loaded,
            "device": "cpu",
            "method": "opencv_a4_calibration",
            "confidence_threshold": processor.confidence_threshold,
            "calibration": {
                "reference_type": "a4_paper",
                "default_pixels_per_cm": round(processor._default_pixels_per_cm, 2),
            },
            "supported_references": [
                {"type": "a4_paper", "size": "21.0 x 29.7 cm"},
            ],
        }

    except Exception as e:
        return {"ready": False, "error": str(e)}


@router.post(
    "/warmup",
    summary="Warm up processor",
    description="Pre-initialize processor for faster first inference",
)
async def warmup_models():
    """Pre-initialize processor"""

    try:
        processor = get_processor()

        # Create a dummy image to test processing
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some test shapes
        cv2.rectangle(dummy_image, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.rectangle(dummy_image, (350, 150), (450, 250), (200, 200, 200), -1)

        # Process to warm up
        processor.process_frame(dummy_image, return_annotated=False)

        return {"success": True, "message": "Processor ready"}

    except Exception as e:
        logger.error(f"Warmup error: {e}")
        return {"success": False, "message": f"Warmup failed: {str(e)}"}
