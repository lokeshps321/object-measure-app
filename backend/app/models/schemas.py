"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class MeasuredObjectResponse(BaseModel):
    """Single measured object data"""

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
    """Response from measurement endpoint"""

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
                    {
                        "width_cm": 12.0,
                        "height_cm": 7.5,
                        "bounding_box": [400, 200, 360, 225],
                    },
                ],
                "processed_image": "base64_encoded_string...",
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")

    class Config:
        json_schema_extra = {"example": {"status": "healthy", "version": "1.0.0"}}


class ErrorResponse(BaseModel):
    """Error response"""

    success: bool = Field(default=False)
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
