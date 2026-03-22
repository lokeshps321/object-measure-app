"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class MeasuredObjectResponse(BaseModel):
    """Single measured object data"""

    object_id: int = Field(default=0, description="Object ID")
    object_type: str = Field(default="2D", description="2D or 3D")
    label: str = Field(default="Object", description="Object label")
    confidence: float = Field(default=0.5, description="Detection confidence")
    length_cm: float = Field(default=0, description="Length in centimeters")
    width_cm: float = Field(default=0, description="Width in centimeters")
    height_cm: Optional[float] = Field(None, description="Height in cm (3D only)")
    bounding_box: Tuple[int, int, int, int] = Field(
        ..., description="(x, y, width, height)"
    )
    center: Optional[Tuple[int, int]] = Field(None, description="Center point")

    class Config:
        json_schema_extra = {
            "example": {
                "object_id": 1,
                "object_type": "3D",
                "label": "Object 1",
                "confidence": 0.85,
                "length_cm": 8.5,
                "width_cm": 5.4,
                "height_cm": 2.1,
                "bounding_box": [100, 150, 255, 162],
                "center": [227, 231],
            }
        }


class MeasurementResponse(BaseModel):
    """Response from measurement endpoint"""

    success: bool = Field(..., description="Whether measurement was successful")
    message: str = Field(..., description="Status message")
    reference_detected: bool = Field(
        default=False, description="Whether reference was detected"
    )
    objects: List[MeasuredObjectResponse] = Field(
        default=[], description="List of measured objects"
    )
    processed_image: Optional[str] = Field(
        None, description="Base64 encoded result image"
    )
    mode: str = Field(default="2d", description="Measurement mode")
    calibration_info: Optional[dict] = Field(
        None, description="Calibration details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Measured 2 object(s)",
                "reference_detected": True,
                "objects": [],
                "processed_image": "base64...",
                "mode": "3d",
            }
        }


class MeasurementRequest(BaseModel):
    """Request body for measurement"""
    image: str = Field(..., description="Base64 encoded image")
    mode: str = Field(default="2d", description="Measurement mode: 2d or 3d")
    camera_distance_cm: float = Field(default=30.0, description="Camera distance in cm")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {"example": {"status": "healthy", "version": "2.0.0"}}


class ErrorResponse(BaseModel):
    """Error response"""

    success: bool = Field(default=False)
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
