"""
Real-time Object Measurement Processor - V3
Accurate measurements using reference object calibration

Key improvements:
1. Credit card / A4 paper reference detection for accurate calibration
2. Proper pixel-to-cm conversion based on reference
3. Perspective correction using detected reference corners
4. Shadow and contour analysis for 3D height estimation
"""

import cv2
import numpy as np
import base64
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ObjectType(str, Enum):
    OBJECT_2D = "2D"
    OBJECT_3D = "3D"


class ReferenceType(str, Enum):
    CREDIT_CARD = "credit_card"  # 8.56 x 5.398 cm (ISO/IEC 7810 ID-1)
    A4_PAPER = "a4_paper"  # 21.0 x 29.7 cm
    CUSTOM = "custom"
    NONE = "none"


# Standard reference object sizes in cm
REFERENCE_SIZES = {
    ReferenceType.CREDIT_CARD: (8.56, 5.398),  # width x height
    ReferenceType.A4_PAPER: (21.0, 29.7),
}


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
class CalibrationData:
    """Calibration data from reference object"""

    pixels_per_cm: float
    reference_type: ReferenceType
    reference_width_cm: float
    reference_height_cm: float
    reference_detected: bool
    homography_matrix: Optional[np.ndarray] = None


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
    calibration_info: Optional[Dict] = None


class RealtimeProcessor:
    """
    Accurate real-time processor for object measurement
    Uses reference object calibration for precise measurements
    """

    # Detection parameters
    MIN_CONTOUR_AREA = 500
    MAX_CONTOUR_AREA = 800000

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize the processor"""
        self.confidence_threshold = confidence_threshold
        self._models_loaded = True

        # Calibration state
        self._calibration: Optional[CalibrationData] = None
        self._default_pixels_per_cm = 37.8  # Default ~96 DPI at 1 inch = 2.54cm
        self._reference_type = ReferenceType.CREDIT_CARD
        self._custom_reference_size = (8.56, 5.398)  # Default to credit card

        logger.info("RealtimeProcessor V3 initialized - Reference-based calibration")

    def set_reference_type(
        self,
        ref_type: ReferenceType,
        custom_width_cm: float = None,
        custom_height_cm: float = None,
    ):
        """Set the reference object type for calibration"""
        self._reference_type = ref_type
        if ref_type == ReferenceType.CUSTOM and custom_width_cm and custom_height_cm:
            self._custom_reference_size = (custom_width_cm, custom_height_cm)
        logger.info(f"Reference type set to: {ref_type}")

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def _detect_reference_object(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect reference object (credit card or A4 paper) in the image
        Returns: (corner_points, confidence)
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)

        # Try multiple edge detection approaches
        best_rect = None
        best_score = 0

        for canny_low, canny_high in [(30, 100), (50, 150), (75, 200)]:
            edges = cv2.Canny(blurred, canny_low, canny_high)

            # Dilate to close gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Looking for 4-sided polygons
                if len(approx) == 4:
                    area = cv2.contourArea(approx)

                    # Reference should be reasonable size (2% to 60% of image)
                    min_area = width * height * 0.02
                    max_area = width * height * 0.6

                    if min_area < area < max_area:
                        # Check if it's roughly rectangular
                        rect = self._order_points(approx.reshape(4, 2))

                        # Calculate aspect ratio
                        w1 = np.linalg.norm(rect[0] - rect[1])
                        w2 = np.linalg.norm(rect[2] - rect[3])
                        h1 = np.linalg.norm(rect[0] - rect[3])
                        h2 = np.linalg.norm(rect[1] - rect[2])

                        avg_w = (w1 + w2) / 2
                        avg_h = (h1 + h2) / 2

                        if avg_w < avg_h:
                            avg_w, avg_h = avg_h, avg_w

                        aspect_ratio = avg_w / avg_h if avg_h > 0 else 0

                        # Check if aspect ratio matches expected reference
                        if self._reference_type == ReferenceType.CREDIT_CARD:
                            expected_ratio = 8.56 / 5.398  # ~1.586
                            ratio_tolerance = 0.25
                        elif self._reference_type == ReferenceType.A4_PAPER:
                            expected_ratio = 29.7 / 21.0  # ~1.414
                            ratio_tolerance = 0.2
                        else:
                            expected_ratio = (
                                self._custom_reference_size[0]
                                / self._custom_reference_size[1]
                            )
                            ratio_tolerance = 0.3

                        ratio_diff = abs(aspect_ratio - expected_ratio)
                        if ratio_diff < ratio_tolerance:
                            # Score based on rectangularity and aspect ratio match
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = area / hull_area if hull_area > 0 else 0

                            # Prefer smaller rectangles that match reference size
                            # Credit card at typical distances: 5-20% of image area
                            # A4 paper: 10-40% of image area
                            if self._reference_type == ReferenceType.CREDIT_CARD:
                                ideal_area_ratio = 0.08  # ~8% of image
                                area_tolerance = 0.15
                            else:  # A4 or custom
                                ideal_area_ratio = 0.25  # ~25% of image
                                area_tolerance = 0.25

                            area_ratio = area / (width * height)
                            area_diff = abs(area_ratio - ideal_area_ratio)
                            area_score = max(0, 1 - area_diff / area_tolerance)

                            # Combined score: rectangularity + aspect ratio match + size match
                            ratio_score = 1 - ratio_diff / ratio_tolerance
                            score = (
                                solidity * 0.3 + ratio_score * 0.4 + area_score * 0.3
                            )

                            if score > best_score:
                                best_score = score
                                best_rect = rect

        return best_rect, best_score

    def _calibrate_from_reference(self, image: np.ndarray) -> CalibrationData:
        """
        Calibrate pixel-to-cm ratio from detected reference object
        """
        ref_points, confidence = self._detect_reference_object(image)

        if ref_points is not None and confidence > 0.2:
            # Get reference size
            if self._reference_type == ReferenceType.CUSTOM:
                ref_width_cm, ref_height_cm = self._custom_reference_size
            else:
                ref_width_cm, ref_height_cm = REFERENCE_SIZES.get(
                    self._reference_type, REFERENCE_SIZES[ReferenceType.CREDIT_CARD]
                )

            # Calculate pixel dimensions of reference
            w1 = np.linalg.norm(ref_points[0] - ref_points[1])
            w2 = np.linalg.norm(ref_points[2] - ref_points[3])
            h1 = np.linalg.norm(ref_points[0] - ref_points[3])
            h2 = np.linalg.norm(ref_points[1] - ref_points[2])

            ref_width_px = (w1 + w2) / 2
            ref_height_px = (h1 + h2) / 2

            # Ensure width > height (landscape orientation)
            if ref_width_px < ref_height_px:
                ref_width_px, ref_height_px = ref_height_px, ref_width_px

            # Calculate pixels per cm (average of both dimensions)
            px_per_cm_w = ref_width_px / ref_width_cm
            px_per_cm_h = ref_height_px / ref_height_cm
            pixels_per_cm = (px_per_cm_w + px_per_cm_h) / 2

            # Calculate homography for perspective correction
            dst_w = int(ref_width_cm * pixels_per_cm)
            dst_h = int(ref_height_cm * pixels_per_cm)
            dst_points = np.array(
                [[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32
            )

            # Ensure proper orientation of reference points
            if ref_width_px < ref_height_px:
                ref_points = np.array(
                    [ref_points[0], ref_points[3], ref_points[2], ref_points[1]]
                )

            homography, _ = cv2.findHomography(ref_points, dst_points)

            self._calibration = CalibrationData(
                pixels_per_cm=pixels_per_cm,
                reference_type=self._reference_type,
                reference_width_cm=ref_width_cm,
                reference_height_cm=ref_height_cm,
                reference_detected=True,
                homography_matrix=homography,
            )

            logger.info(f"Calibration successful: {pixels_per_cm:.2f} px/cm")
            return self._calibration

        # No reference detected - use default or previous calibration
        if self._calibration is None:
            self._calibration = CalibrationData(
                pixels_per_cm=self._default_pixels_per_cm,
                reference_type=ReferenceType.NONE,
                reference_width_cm=0,
                reference_height_cm=0,
                reference_detected=False,
            )

        return self._calibration

    def _detect_objects(
        self, image: np.ndarray, calibration: CalibrationData
    ) -> List[dict]:
        """
        Detect and measure objects in the image
        """
        height, width = image.shape[:2]
        detected_objects = []

        # Convert to different color spaces for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Multi-scale edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Find contours
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        object_id = 0
        for contour in contours[:10]:  # Process top 10 largest
            area = cv2.contourArea(contour)

            if area < self.MIN_CONTOUR_AREA or area > self.MAX_CONTOUR_AREA:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter very thin objects
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
            if aspect_ratio > 15:
                continue

            # Filter objects touching image edge
            margin = 5
            if (
                x < margin
                or y < margin
                or x + w > width - margin
                or y + h > height - margin
            ):
                continue

            # Calculate confidence
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Circularity check
            circularity = (
                4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            )

            # Confidence based on shape regularity
            confidence = min(0.95, solidity * 0.6 + circularity * 0.2 + 0.2)

            if confidence < self.confidence_threshold:
                continue

            object_id += 1

            # Get rotated rectangle for more accurate dimensions
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Dimensions from rotated rect
            rect_w, rect_h = rect[1]
            if rect_w < rect_h:
                rect_w, rect_h = rect_h, rect_w

            # Calculate center
            cx = int(rect[0][0])
            cy = int(rect[0][1])

            # Analyze ROI for 3D detection
            roi = gray[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]

            # 3D detection based on multiple factors
            is_3d = False
            depth_factor = 0.0

            if roi.size > 0:
                # Texture variance (complex textures suggest 3D)
                texture_variance = np.var(roi)

                # Gradient analysis (strong gradients suggest 3D surfaces)
                sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                gradient_mean = np.mean(gradient_magnitude)

                # Edge density within object
                roi_edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(roi_edges > 0) / roi.size if roi.size > 0 else 0

                # Shadow detection (darker regions at edges suggest 3D)
                h_third = roi.shape[0] // 3
                if h_third > 0:
                    top_brightness = np.mean(roi[:h_third, :])
                    bottom_brightness = np.mean(roi[-h_third:, :])
                    brightness_diff = abs(top_brightness - bottom_brightness)
                else:
                    brightness_diff = 0

                # Combine factors for 3D detection
                texture_score = min(1.0, texture_variance / 2000)
                gradient_score = min(1.0, gradient_mean / 50)
                edge_score = min(1.0, edge_density * 20)
                shadow_score = min(1.0, brightness_diff / 50)

                depth_factor = (
                    texture_score * 0.2
                    + gradient_score * 0.3
                    + edge_score * 0.2
                    + shadow_score * 0.3
                )

                # 3D if depth factor is significant or complex shape
                is_3d = depth_factor > 0.4 or len(approx) > 6 or solidity < 0.85

            detected_objects.append(
                {
                    "id": object_id,
                    "bbox": (x, y, w, h),
                    "rotated_rect": rect,
                    "box_points": box,
                    "rect_width": rect_w,
                    "rect_height": rect_h,
                    "center": (cx, cy),
                    "confidence": confidence,
                    "contour": contour,
                    "is_3d": is_3d,
                    "depth_factor": depth_factor,
                    "num_vertices": len(approx),
                }
            )

        return detected_objects

    def _calculate_dimensions(
        self, obj: dict, calibration: CalibrationData, image_height: int
    ) -> Tuple[float, float, Optional[float]]:
        """
        Calculate real-world dimensions using calibration data
        """
        # Use rotated rectangle dimensions for accuracy
        width_px = obj["rect_width"]
        height_px = obj["rect_height"]

        # Get pixels per cm from calibration
        px_per_cm = calibration.pixels_per_cm

        # If no reference detected, apply distance-based estimation
        if not calibration.reference_detected:
            # Estimate based on typical phone camera at ~30cm
            # Assume 12MP camera (4000x3000) at 30cm gives ~40 px/cm
            y = obj["center"][1]
            # Objects lower in image are typically closer
            position_factor = 1.0 + (y / image_height) * 0.15
            px_per_cm = self._default_pixels_per_cm * position_factor

        # Calculate dimensions
        length_cm = round(width_px / px_per_cm, 1)
        breadth_cm = round(height_px / px_per_cm, 1)

        # Ensure length >= breadth
        if length_cm < breadth_cm:
            length_cm, breadth_cm = breadth_cm, length_cm

        # Minimum size
        length_cm = max(0.5, length_cm)
        breadth_cm = max(0.5, breadth_cm)

        # Calculate height for 3D objects
        height_cm = None
        if obj["is_3d"]:
            # Estimate height based on depth factor and smaller dimension
            depth_factor = obj["depth_factor"]

            # Height estimation: use depth factor to scale
            # Typical objects have height between 20% to 80% of their smallest dimension
            min_dim = min(length_cm, breadth_cm)
            height_cm = round(min_dim * (0.2 + depth_factor * 0.6), 1)
            height_cm = max(
                0.5, min(height_cm, min_dim * 1.5)
            )  # Cap at 150% of min dimension

        return length_cm, breadth_cm, height_cm

    def _annotate_image(
        self,
        image: np.ndarray,
        objects: List[MeasuredObject3D],
        calibration: CalibrationData,
        ref_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw measurement annotations on image"""
        annotated = image.copy()

        # Draw reference object if detected
        if ref_points is not None:
            pts = ref_points.astype(np.int32)
            cv2.polylines(annotated, [pts], True, (255, 255, 0), 2)

            # Label reference
            ref_label = f"Reference: {calibration.reference_type.value}"
            cv2.putText(
                annotated,
                ref_label,
                (int(pts[0][0]), int(pts[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        # Draw calibration status
        if calibration.reference_detected:
            status = f"Calibrated: {calibration.pixels_per_cm:.1f} px/cm"
            status_color = (0, 255, 0)
        else:
            status = "No reference - estimates only"
            status_color = (0, 165, 255)

        cv2.putText(
            annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
        )

        for obj in objects:
            x, y, w, h = obj.bounding_box

            # Colors
            if obj.object_type == ObjectType.OBJECT_3D:
                color = (0, 255, 0)  # Green for 3D
                text_bg = (0, 180, 0)
            else:
                color = (255, 165, 0)  # Blue for 2D (BGR)
                text_bg = (200, 130, 0)

            # Draw bounding box with rounded corners effect
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw corner markers
            marker_len = min(20, w // 4, h // 4)
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

            # Prepare labels
            type_label = f"{obj.object_type.value}: {obj.label}"
            if obj.object_type == ObjectType.OBJECT_3D:
                dim_text = f"L:{obj.length_cm} B:{obj.breadth_cm} H:{obj.height_cm} cm"
            else:
                dim_text = f"L:{obj.length_cm} B:{obj.breadth_cm} cm"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Draw label at top
            (tw, th), _ = cv2.getTextSize(type_label, font, font_scale, thickness)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 8, y), text_bg, -1)
            cv2.putText(
                annotated,
                type_label,
                (x + 4, y - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # Draw dimensions at bottom
            (dw, dh), _ = cv2.getTextSize(dim_text, font, font_scale, thickness)
            cv2.rectangle(
                annotated, (x, y + h), (x + dw + 8, y + h + dh + 8), text_bg, -1
            )
            cv2.putText(
                annotated,
                dim_text,
                (x + 4, y + h + dh + 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # Draw dimension arrows
            arrow_color = (255, 255, 255)
            mid_y = y + h // 2
            mid_x = x + w // 2

            # Horizontal arrow
            if w > 40:
                cv2.arrowedLine(
                    annotated,
                    (x + 5, mid_y),
                    (x + w - 5, mid_y),
                    arrow_color,
                    2,
                    tipLength=0.05,
                )

            # Vertical arrow
            if h > 40:
                cv2.arrowedLine(
                    annotated,
                    (mid_x, y + 5),
                    (mid_x, y + h - 5),
                    arrow_color,
                    2,
                    tipLength=0.05,
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
            # Step 1: Calibrate from reference object
            calibration = self._calibrate_from_reference(image)

            # Also get reference points for annotation
            ref_points, _ = self._detect_reference_object(image)

            # Step 2: Detect objects
            detected = self._detect_objects(image, calibration)

            if not detected:
                return RealtimeMeasurementResult(
                    success=True,
                    message="No objects detected. Place objects on contrasting background.",
                    objects=[],
                    frame_width=width,
                    frame_height=height,
                    processing_time_ms=round((time.time() - start_time) * 1000, 2),
                    calibration_info={
                        "reference_detected": calibration.reference_detected,
                        "reference_type": calibration.reference_type.value,
                        "pixels_per_cm": round(calibration.pixels_per_cm, 2),
                    },
                )

            # Step 3: Calculate dimensions for each object
            measured_objects = []
            for obj in detected:
                length_cm, breadth_cm, height_cm = self._calculate_dimensions(
                    obj, calibration, height
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
                    depth_value=round(obj["depth_factor"], 3),
                )
                measured_objects.append(measured)

            # Step 4: Annotate image
            annotated_base64 = None
            if return_annotated:
                annotated = self._annotate_image(
                    image, measured_objects, calibration, ref_points
                )
                _, buffer = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                annotated_base64 = base64.b64encode(buffer).decode("utf-8")

            processing_time = (time.time() - start_time) * 1000

            message = f"Detected {len(measured_objects)} object(s)"
            if calibration.reference_detected:
                message += f" (calibrated with {calibration.reference_type.value})"
            else:
                message += " (estimates - add reference for accuracy)"

            return RealtimeMeasurementResult(
                success=True,
                message=message,
                objects=measured_objects,
                frame_width=width,
                frame_height=height,
                processing_time_ms=round(processing_time, 2),
                annotated_image_base64=annotated_base64,
                calibration_info={
                    "reference_detected": calibration.reference_detected,
                    "reference_type": calibration.reference_type.value,
                    "pixels_per_cm": round(calibration.pixels_per_cm, 2),
                    "reference_width_cm": calibration.reference_width_cm,
                    "reference_height_cm": calibration.reference_height_cm,
                },
            )

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return RealtimeMeasurementResult(
                success=False,
                message=f"Processing error: {str(e)}",
                objects=[],
                frame_width=width,
                frame_height=height,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

    def calibrate(
        self,
        reference_distance_cm: float = 30.0,
        scale_factor: float = 1.0,
        reference_type: str = "credit_card",
        reference_width_cm: float = None,
        reference_height_cm: float = None,
    ):
        """Set calibration parameters manually"""

        # Set reference type
        try:
            ref_type = ReferenceType(reference_type)
        except ValueError:
            ref_type = ReferenceType.CREDIT_CARD

        self._reference_type = ref_type

        if reference_width_cm and reference_height_cm:
            self._custom_reference_size = (reference_width_cm, reference_height_cm)
            self._reference_type = ReferenceType.CUSTOM

        # Adjust default pixels per cm based on distance
        # At 30cm with typical phone camera: ~38 px/cm
        # Inverse relationship with distance
        self._default_pixels_per_cm = (
            38.0 * (30.0 / reference_distance_cm) * scale_factor
        )

        logger.info(
            f"Manual calibration: type={ref_type}, distance={reference_distance_cm}cm, "
            f"default_px_per_cm={self._default_pixels_per_cm:.2f}"
        )


# Global processor instance
_processor: Optional[RealtimeProcessor] = None


def get_processor() -> RealtimeProcessor:
    """Get or create the global processor instance"""
    global _processor
    if _processor is None:
        _processor = RealtimeProcessor(confidence_threshold=0.35)
    return _processor


def process_frame_for_measurement(
    image_data: bytes, return_annotated: bool = True
) -> RealtimeMeasurementResult:
    """Process image bytes for measurement"""
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
