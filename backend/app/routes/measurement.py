"""
API route handlers for measurement endpoints
Supports 2D and 3D measurement - no reference sheet needed
"""

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
import base64
import logging

from app.models.schemas import (
    MeasurementResponse,
    MeasuredObjectResponse,
    MeasurementRequest,
    ErrorResponse,
)
from app.utils.image_processing import measure_objects

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["measurement"])


@router.post(
    "/measure/base64",
    response_model=MeasurementResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image data"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Measure objects from photo",
    description="""
    Measure objects in a photo. No reference sheet needed!
    
    **How to use:**
    1. Take a photo of any object
    2. Specify mode: "2d" for Length×Width, "3d" for Length×Width×Height
    3. Optionally specify camera distance (default 30cm)
    
    **Returns:**
    - Measurements in centimeters for each detected object
    - Annotated image with measurements drawn
    """,
)
async def measure_base64(data: MeasurementRequest):
    """Process base64 encoded image and measure objects"""

    try:
        image_data = data.image
        mode = data.mode or "2d"
        camera_distance_cm = data.camera_distance_cm or 30.0

        # Remove data URL prefix if present
        raw_base64 = image_data
        if "," in image_data:
            raw_base64 = image_data.split(",")[1]

        # Decode base64
        try:
            image_bytes = base64.b64decode(raw_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400, detail="Could not decode image from base64 data"
            )

        # Process the image (tries HF Space first, falls back to local)
        result = measure_objects(
            image=image,
            mode=mode,
            camera_distance_cm=camera_distance_cm,
            image_base64=raw_base64,
        )

        # Build response
        measured_objects = [
            MeasuredObjectResponse(
                object_id=obj.object_id,
                object_type=obj.object_type,
                label=obj.label,
                confidence=obj.confidence,
                length_cm=obj.length_cm,
                width_cm=obj.width_cm,
                height_cm=obj.height_cm,
                bounding_box=obj.bounding_box,
                center=obj.center,
            )
            for obj in result.objects
        ]

        return MeasurementResponse(
            success=result.success,
            message=result.message,
            reference_detected=result.reference_detected,
            objects=measured_objects,
            processed_image=result.processed_image_base64,
            mode=result.mode,
            calibration_info=result.calibration_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Legacy endpoint - same as above but accepts raw dict
@router.post("/measure/legacy")
async def measure_legacy(data: dict):
    """Legacy endpoint for backwards compatibility"""
    request = MeasurementRequest(
        image=data.get("image", ""),
        mode=data.get("mode", "2d"),
        camera_distance_cm=data.get("camera_distance_cm", 30.0),
    )
    return await measure_base64(request)
