"""
Real-time Object Measurement Processor (Lightweight Version)
Uses OpenCV for object detection and measurement
Works without heavy ML dependencies (no PyTorch/YOLO)
"""

import cv2
import numpy as np
import base64
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ObjectType(str, Enum):
    OBJECT_2D = "2D"
    OBJECT_3D = "3D"


@dataclass
class MeasuredObject3D:
    """Represents a measured object with 2D or 3D dimensions"""

    object_id: int
    object_type: ObjectType
    label: str
    confidence: float
    length_cm: float
    breadth_cm: float
    height_cm: Optional[float]
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    depth_value: float


@dataclass
class RealtimeMeasurementResult:
    """Result from real-time measurement"""

    success: bool
    message: str
    objects: List[MeasuredObject3D]
    frame_width: int
    frame_height: int
    processing_time_ms: float
    annotated_image_base64: Optional[str] = None


class RealtimeProcessor:
    """
    Lightweight real-time processor for object measurement using OpenCV
    No heavy ML dependencies - works on Render free tier
    """

    # Calibration: pixels to cm at reference distance
    # Assumes phone camera at ~30cm distance from object
    PIXELS_PER_CM = 15.0  # Approximate - can be calibrated
    REFERENCE_DISTANCE_CM = 30.0

    # Detection parameters
    MIN_CONTOUR_AREA = 1000
    MAX_CONTOUR_AREA = 500000

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize the processor"""
        self.confidence_threshold = confidence_threshold
        self._scale_factor = 1.0
        self._models_loaded = True  # No models to load in lightweight version
        logger.info("Lightweight RealtimeProcessor initialized (OpenCV only)")

    def _detect_objects_opencv(self, image: np.ndarray) -> List[dict]:
        """
        Detect objects using OpenCV contour detection
        """
        height, width = image.shape[:2]
        detected_objects = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # Dilate to close gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and process contours
        object_id = 0
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.MIN_CONTOUR_AREA or area > self.MAX_CONTOUR_AREA:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out very thin objects (likely edges)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
            if aspect_ratio > 10:
                continue

            # Filter objects touching the edge
            margin = 10
            if (
                x < margin
                or y < margin
                or x + w > width - margin
                or y + h > height - margin
            ):
                continue

            # Calculate confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Higher confidence for more regular shapes
            confidence = min(0.95, solidity * 0.8 + 0.2)

            if confidence < self.confidence_threshold:
                continue

            object_id += 1

            # Determine if likely 3D based on shape analysis
            # Use moment analysis to detect depth perception cues
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Estimate 3D-ness based on shape complexity and texture variance
            roi = gray[y : y + h, x : x + w]
            texture_variance = np.var(roi) if roi.size > 0 else 0

            # Objects with high texture variance are more likely 3D
            is_3d = texture_variance > 1000 or len(approx) > 6

            detected_objects.append(
                {
                    "id": object_id,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "confidence": confidence,
                    "contour": contour,
                    "is_3d": is_3d,
                    "texture_variance": texture_variance,
                    "num_vertices": len(approx),
                }
            )

        return detected_objects

    def _calculate_dimensions(
        self,
        bbox: Tuple[int, int, int, int],
        is_3d: bool,
        texture_variance: float,
        image_height: int,
    ) -> Tuple[float, float, Optional[float]]:
        """
        Calculate real-world dimensions from bounding box
        """
        x, y, w, h = bbox

        # Calculate scale based on position in image
        # Objects lower in image are usually closer
        position_factor = 1.0 + (y / image_height) * 0.3

        # Apply scale factor
        scale = self.PIXELS_PER_CM * self._scale_factor * position_factor

        # Calculate length and breadth
        length_cm = round(w / scale, 1)
        breadth_cm = round(h / scale, 1)

        # Ensure minimum size
        length_cm = max(1.0, length_cm)
        breadth_cm = max(1.0, breadth_cm)

        # Calculate height for 3D objects
        height_cm = None
        if is_3d:
            # Estimate height based on texture variance and shape
            # Higher variance suggests more depth
            depth_factor = min(1.0, texture_variance / 3000)
            height_cm = round(
                min(length_cm, breadth_cm) * (0.3 + depth_factor * 0.5), 1
            )
            height_cm = max(1.0, height_cm)

        return length_cm, breadth_cm, height_cm

    def _annotate_image(
        self, image: np.ndarray, objects: List[MeasuredObject3D]
    ) -> np.ndarray:
        """Draw measurement annotations on image"""
        annotated = image.copy()

        for obj in objects:
            x, y, w, h = obj.bounding_box

            # Colors
            if obj.object_type == ObjectType.OBJECT_3D:
                color = (0, 255, 0)  # Green for 3D
                text_bg = (0, 180, 0)
            else:
                color = (255, 165, 0)  # Orange for 2D
                text_bg = (200, 130, 0)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

            # Draw corner markers
            marker_len = 15
            thickness = 3
            # Top-left
            cv2.line(annotated, (x, y), (x + marker_len, y), color, thickness)
            cv2.line(annotated, (x, y), (x, y + marker_len), color, thickness)
            # Top-right
            cv2.line(annotated, (x + w, y), (x + w - marker_len, y), color, thickness)
            cv2.line(annotated, (x + w, y), (x + w, y + marker_len), color, thickness)
            # Bottom-left
            cv2.line(annotated, (x, y + h), (x + marker_len, y + h), color, thickness)
            cv2.line(annotated, (x, y + h), (x, y + h - marker_len), color, thickness)
            # Bottom-right
            cv2.line(
                annotated, (x + w, y + h), (x + w - marker_len, y + h), color, thickness
            )
            cv2.line(
                annotated, (x + w, y + h), (x + w, y + h - marker_len), color, thickness
            )

            # Prepare text
            type_label = f"{obj.object_type.value}: {obj.label}"
            if obj.object_type == ObjectType.OBJECT_3D:
                dim_text = f"L:{obj.length_cm} B:{obj.breadth_cm} H:{obj.height_cm} cm"
            else:
                dim_text = f"L:{obj.length_cm} B:{obj.breadth_cm} cm"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Draw label background and text at top
            (tw, th), _ = cv2.getTextSize(type_label, font, font_scale, thickness)
            cv2.rectangle(annotated, (x, y - th - 10), (x + tw + 10, y), text_bg, -1)
            cv2.putText(
                annotated,
                type_label,
                (x + 5, y - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # Draw dimensions at bottom
            (dw, dh), _ = cv2.getTextSize(dim_text, font, font_scale, thickness)
            cv2.rectangle(
                annotated, (x, y + h), (x + dw + 10, y + h + dh + 10), text_bg, -1
            )
            cv2.putText(
                annotated,
                dim_text,
                (x + 5, y + h + dh + 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # Draw dimension arrows
            arrow_color = (255, 255, 255)
            # Horizontal arrow (Length)
            mid_y = y + h // 2
            cv2.arrowedLine(
                annotated,
                (x + 5, mid_y),
                (x + w - 5, mid_y),
                arrow_color,
                2,
                tipLength=0.03,
            )
            # Vertical arrow (Breadth)
            mid_x = x + w // 2
            cv2.arrowedLine(
                annotated,
                (mid_x, y + 5),
                (mid_x, y + h - 5),
                arrow_color,
                2,
                tipLength=0.03,
            )

        return annotated

    def process_frame(
        self, image: np.ndarray, return_annotated: bool = True
    ) -> RealtimeMeasurementResult:
        """
        Process a single frame and measure objects
        """
        start_time = time.time()
        height, width = image.shape[:2]

        try:
            # Detect objects using OpenCV
            detected = self._detect_objects_opencv(image)

            if not detected:
                return RealtimeMeasurementResult(
                    success=True,
                    message="No objects detected. Try better lighting or clearer background.",
                    objects=[],
                    frame_width=width,
                    frame_height=height,
                    processing_time_ms=round((time.time() - start_time) * 1000, 2),
                )

            # Convert to measured objects
            measured_objects = []
            for obj in detected:
                length_cm, breadth_cm, height_cm = self._calculate_dimensions(
                    obj["bbox"], obj["is_3d"], obj["texture_variance"], height
                )

                measured = MeasuredObject3D(
                    object_id=obj["id"],
                    object_type=ObjectType.OBJECT_3D
                    if obj["is_3d"]
                    else ObjectType.OBJECT_2D,
                    label=f"Object {obj['id']}",
                    confidence=round(obj["confidence"], 2),
                    length_cm=length_cm,
                    breadth_cm=breadth_cm,
                    height_cm=height_cm,
                    bounding_box=obj["bbox"],
                    center=obj["center"],
                    depth_value=obj["texture_variance"] / 1000,
                )
                measured_objects.append(measured)

            # Annotate image if requested
            annotated_base64 = None
            if return_annotated:
                annotated = self._annotate_image(image, measured_objects)
                _, buffer = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                annotated_base64 = base64.b64encode(buffer).decode("utf-8")

            processing_time = (time.time() - start_time) * 1000

            return RealtimeMeasurementResult(
                success=True,
                message=f"Detected {len(measured_objects)} object(s)",
                objects=measured_objects,
                frame_width=width,
                frame_height=height,
                processing_time_ms=round(processing_time, 2),
                annotated_image_base64=annotated_base64,
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return RealtimeMeasurementResult(
                success=False,
                message=f"Processing error: {str(e)}",
                objects=[],
                frame_width=width,
                frame_height=height,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

    def calibrate(self, reference_distance_cm: float = 30.0, scale_factor: float = 1.0):
        """Set calibration parameters"""
        self.REFERENCE_DISTANCE_CM = reference_distance_cm
        self._scale_factor = scale_factor
        logger.info(
            f"Calibration: distance={reference_distance_cm}cm, scale={scale_factor}"
        )


# Global processor instance
_processor: Optional[RealtimeProcessor] = None


def get_processor() -> RealtimeProcessor:
    """Get or create the global processor instance"""
    global _processor
    if _processor is None:
        _processor = RealtimeProcessor(confidence_threshold=0.4)
    return _processor


def process_frame_for_measurement(
    image_data: bytes, return_annotated: bool = True
) -> RealtimeMeasurementResult:
    """
    Process image bytes for measurement
    """
    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return RealtimeMeasurementResult(
            success=False,
            message="Could not decode image",
            objects=[],
            frame_width=0,
            frame_height=0,
            processing_time_ms=0,
        )

    processor = get_processor()
    return processor.process_frame(image, return_annotated)
