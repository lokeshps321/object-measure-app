"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from enum import Enum


# ============== Legacy Schemas (for backward compatibility) ==============


class MeasuredObjectResponse(BaseModel):
    """Single measured object data (legacy 2D only)"""

    width_cm: float = Field(..., description="Width in centimeters")
    height_cm: float = Field(..., description="Height in centimeters")
    bounding_box: Tuple[int, int, int, int] = Field(
        ..., description="(x, y, width, height)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "width_cm": 8.5,
                "height_cm": 5.4,
                "bounding_box": [100, 150, 255, 162],
            }
        }


class MeasurementResponse(BaseModel):
    """Response from measurement endpoint (legacy)"""

    success: bool = Field(..., description="Whether measurement was successful")
    message: str = Field(..., description="Status message")
    reference_detected: bool = Field(
        ..., description="Whether A4 reference was detected"
    )
    objects: List[MeasuredObjectResponse] = Field(
        default=[], description="List of measured objects"
    )
    processed_image: Optional[str] = Field(
        None, description="Base64 encoded result image"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully measured 2 object(s)",
                "reference_detected": True,
                "objects": [
                    {
                        "width_cm": 8.5,
                        "height_cm": 5.4,
                        "bounding_box": [100, 150, 255, 162],
                    },
                ],
                "processed_image": "base64_encoded_string...",
            }
        }


# ============== New Realtime 2D/3D Schemas ==============


class ObjectType(str, Enum):
    """Type of detected object"""

    OBJECT_2D = "2D"
    OBJECT_3D = "3D"


class MeasuredObject3DResponse(BaseModel):
    """Single measured object with 2D or 3D dimensions"""

    object_id: int = Field(..., description="Unique object ID in frame")
    object_type: ObjectType = Field(..., description="2D or 3D object type")
    label: str = Field(..., description="Detected object label (e.g., 'book', 'box')")
    confidence: float = Field(..., description="Detection confidence (0-1)")

    # Dimensions
    length_cm: float = Field(..., description="Length in centimeters")
    breadth_cm: float = Field(..., description="Breadth in centimeters")
    height_cm: Optional[float] = Field(
        None, description="Height in cm (only for 3D objects)"
    )

    # Position
    bounding_box: Tuple[int, int, int, int] = Field(
        ..., description="(x, y, width, height) in pixels"
    )
    center: Tuple[int, int] = Field(..., description="Center point (x, y)")
    depth_value: float = Field(..., description="Relative depth value")

    class Config:
        json_schema_extra = {
            "example": {
                "object_id": 1,
                "object_type": "3D",
                "label": "box",
                "confidence": 0.92,
                "length_cm": 15.5,
                "breadth_cm": 10.2,
                "height_cm": 8.0,
                "bounding_box": [100, 150, 200, 130],
                "center": [200, 215],
                "depth_value": 0.65,
            }
        }


class RealtimeMeasurementResponse(BaseModel):
    """Response from real-time measurement endpoint"""

    success: bool = Field(..., description="Whether processing was successful")
    message: str = Field(..., description="Status message")
    objects: List[MeasuredObject3DResponse] = Field(
        default=[], description="List of measured objects"
    )
    frame_width: int = Field(..., description="Input frame width")
    frame_height: int = Field(..., description="Input frame height")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    annotated_image: Optional[str] = Field(
        None, description="Base64 encoded annotated image"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Measured 2 object(s)",
                "objects": [
                    {
                        "object_id": 1,
                        "object_type": "3D",
                        "label": "box",
                        "confidence": 0.92,
                        "length_cm": 15.5,
                        "breadth_cm": 10.2,
                        "height_cm": 8.0,
                        "bounding_box": [100, 150, 200, 130],
                        "center": [200, 215],
                        "depth_value": 0.65,
                    },
                    {
                        "object_id": 2,
                        "object_type": "2D",
                        "label": "book",
                        "confidence": 0.88,
                        "length_cm": 21.0,
                        "breadth_cm": 29.7,
                        "height_cm": None,
                        "bounding_box": [350, 200, 250, 350],
                        "center": [475, 375],
                        "depth_value": 0.52,
                    },
                ],
                "frame_width": 1920,
                "frame_height": 1080,
                "processing_time_ms": 156.5,
                "annotated_image": "base64_encoded_string...",
            }
        }


class RealtimeMeasurementRequest(BaseModel):
    """Request body for real-time measurement"""

    image: str = Field(..., description="Base64 encoded image")
    return_annotated: bool = Field(
        default=True, description="Whether to return annotated image"
    )
    calibration_distance_cm: Optional[float] = Field(
        None, description="Distance from camera to objects in cm (for calibration)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image": "base64_encoded_image_data...",
                "return_annotated": True,
                "calibration_distance_cm": 100.0,
            }
        }


class CalibrationRequest(BaseModel):
    """Request to calibrate measurement system"""

    reference_distance_cm: float = Field(
        default=100.0, description="Distance from camera to reference object"
    )
    reference_object_width_cm: Optional[float] = Field(
        None, description="Known width of reference object"
    )
    reference_object_height_cm: Optional[float] = Field(
        None, description="Known height of reference object"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reference_distance_cm": 100.0,
                "reference_object_width_cm": 21.0,
                "reference_object_height_cm": 29.7,
            }
        }


class CalibrationResponse(BaseModel):
    """Response from calibration endpoint"""

    success: bool
    message: str
    scale_factor: float = Field(..., description="Applied scale factor")


# ============== Common Schemas ==============


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(
        default=False, description="Whether ML models are loaded"
    )

    class Config:
        json_schema_extra = {
            "example": {"status": "healthy", "version": "2.0.0", "models_loaded": True}
        }


class ErrorResponse(BaseModel):
    """Error response"""

    success: bool = Field(default=False)
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
