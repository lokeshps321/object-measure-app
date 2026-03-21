"""
Real-time Object Measurement Processor - V4
Place objects on A4 paper for accurate measurements

How it works:
1. Detect the A4 paper (white rectangle) in the image
2. Use A4 paper size (21.0 x 29.7 cm) to calculate pixels-per-cm
3. Detect all objects ON the A4 paper
4. Measure length, width, and estimate height for 3D objects
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


# A4 paper dimensions in cm
A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7


@dataclass
class MeasuredObject3D:
    """Represents a measured object with dimensions"""

    object_id: int
    object_type: ObjectType
    label: str
    confidence: float
    length_cm: float
    width_cm: float
    height_cm: Optional[float]
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[int, int]


@dataclass
class CalibrationData:
    """Calibration data from A4 paper detection"""

    pixels_per_cm: float
    a4_detected: bool
    a4_corners: Optional[np.ndarray] = None


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
    Measure objects placed on A4 paper
    """

    # Detection parameters
    MIN_OBJECT_AREA = (
        300  # Minimum object size in pixels (lowered for small objects like earbuds)
    )
    MAX_OBJECT_AREA = 500000  # Maximum object size

    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self._models_loaded = True
        self._calibration: Optional[CalibrationData] = None

        # Default pixels per cm (will be overridden by A4 detection)
        # Assume phone camera at ~30cm from A4 paper
        self._default_pixels_per_cm = 40.0

        logger.info("RealtimeProcessor V4 initialized - A4 paper reference")

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def _detect_a4_paper(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect A4 paper (large white/light rectangle) in the image
        Returns: (corner_points, pixels_per_cm)
        """
        height, width = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)

        # Detect white/light regions (A4 paper is usually white)
        # Also try edge detection for paper with objects on it

        best_rect = None
        best_score = 0
        best_pixels_per_cm = self._default_pixels_per_cm

        # Method 1: Threshold for white paper
        _, white_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Method 2: Adaptive threshold
        adaptive_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Method 3: Edge detection
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Try all methods
        for mask in [white_mask, adaptive_mask, edges]:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Looking for 4-sided polygon (rectangle)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)

                    # A4 paper should be large - at least 15% of image, up to 80%
                    min_area = width * height * 0.15
                    max_area = width * height * 0.80

                    if min_area < area < max_area:
                        # Order points
                        rect = self._order_points(approx.reshape(4, 2))

                        # Calculate dimensions
                        w1 = np.linalg.norm(rect[0] - rect[1])
                        w2 = np.linalg.norm(rect[2] - rect[3])
                        h1 = np.linalg.norm(rect[0] - rect[3])
                        h2 = np.linalg.norm(rect[1] - rect[2])

                        avg_w = (w1 + w2) / 2
                        avg_h = (h1 + h2) / 2

                        # Ensure width < height (portrait A4)
                        if avg_w > avg_h:
                            avg_w, avg_h = avg_h, avg_w

                        # A4 aspect ratio is 21/29.7 = 0.707 (portrait)
                        aspect_ratio = avg_w / avg_h if avg_h > 0 else 0
                        expected_ratio = A4_WIDTH_CM / A4_HEIGHT_CM  # 0.707

                        ratio_diff = abs(aspect_ratio - expected_ratio)

                        # Accept if aspect ratio is close to A4
                        if ratio_diff < 0.15:
                            # Check rectangularity
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = area / hull_area if hull_area > 0 else 0

                            # Score based on size, aspect ratio, and rectangularity
                            area_ratio = area / (width * height)
                            size_score = min(1.0, area_ratio / 0.4)  # Prefer larger
                            ratio_score = 1 - ratio_diff / 0.15

                            score = (
                                solidity * 0.3 + ratio_score * 0.4 + size_score * 0.3
                            )

                            if score > best_score and score > 0.5:
                                best_score = score
                                best_rect = rect

                                # Calculate pixels per cm
                                # A4 is 21 x 29.7 cm
                                px_per_cm_w = avg_w / A4_WIDTH_CM
                                px_per_cm_h = avg_h / A4_HEIGHT_CM
                                best_pixels_per_cm = (px_per_cm_w + px_per_cm_h) / 2

        return best_rect, best_pixels_per_cm

    def _detect_objects_on_paper(
        self, image: np.ndarray, a4_corners: Optional[np.ndarray], pixels_per_cm: float
    ) -> List[dict]:
        """
        Detect objects placed on the A4 paper
        Uses multiple detection methods for robustness
        """
        height, width = image.shape[:2]
        detected_objects = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Create mask for A4 paper region if detected
        paper_mask = None
        if a4_corners is not None:
            paper_mask = np.zeros((height, width), dtype=np.uint8)
            pts = a4_corners.astype(np.int32)
            cv2.fillPoly(paper_mask, [pts], 255)

        # Collect contours from multiple methods
        all_contours = []

        # Method 1: Edge detection (good for textured objects)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        # Apply paper_mask to only detect edges within the A4 paper
        if paper_mask is not None:
            edges = cv2.bitwise_and(edges, paper_mask)
        contours1, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours1)

        # Method 2: Threshold for dark objects on white paper
        _, dark_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        if paper_mask is not None:
            dark_mask = cv2.bitwise_and(dark_mask, paper_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours2, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours2)

        # Method 3: Adaptive threshold (good for varying lighting)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        if paper_mask is not None:
            adaptive = cv2.bitwise_and(adaptive, paper_mask)
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours3, _ = cv2.findContours(
            adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours3)

        # Method 4: Color-based detection (objects that differ from white)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Detect non-white areas (saturation > 20 or value < 220)
        color_mask = cv2.inRange(hsv, (0, 20, 0), (180, 255, 255))
        low_value_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 220))
        combined_color = cv2.bitwise_or(color_mask, low_value_mask)
        if paper_mask is not None:
            combined_color = cv2.bitwise_and(combined_color, paper_mask)
        combined_color = cv2.morphologyEx(
            combined_color, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        contours4, _ = cv2.findContours(
            combined_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours4)

        # Method 5: Shadow/gradient detection for white objects on white paper
        # Detect subtle shadows and gradients that indicate 3D objects
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, shadow_mask = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
        if paper_mask is not None:
            shadow_mask = cv2.bitwise_and(shadow_mask, paper_mask)
        shadow_mask = cv2.morphologyEx(
            shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=3
        )
        shadow_mask = cv2.morphologyEx(
            shadow_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        contours5, _ = cv2.findContours(
            shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours5)

        # Sort all contours by area and remove duplicates
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

        # Track detected regions to avoid duplicates
        detected_regions = []

        object_id = 0
        for contour in all_contours[:30]:  # Check top 30 from all methods
            area = cv2.contourArea(contour)

            if area < self.MIN_OBJECT_AREA or area > self.MAX_OBJECT_AREA:
                continue

            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)

            # Skip very thin objects
            aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 100
            if aspect > 12:
                continue

            # Skip objects at image edge
            margin = 5
            if (
                x < margin
                or y < margin
                or x + w > width - margin
                or y + h > height - margin
            ):
                continue

            # Check if object is on the A4 paper (if detected)
            cx = x + w // 2
            cy = y + h // 2

            if paper_mask is not None:
                # Object center should be on paper
                if paper_mask[cy, cx] == 0:
                    continue

                # Skip if this IS the A4 paper itself
                paper_area = cv2.contourArea(
                    a4_corners.astype(np.int32).reshape(-1, 1, 2)
                )
                if abs(area - paper_area) / paper_area < 0.15:
                    continue

            # Check for duplicate regions (from multiple detection methods)
            is_duplicate = False
            for rx, ry, rw, rh in detected_regions:
                # Check overlap
                overlap_x = max(0, min(x + w, rx + rw) - max(x, rx))
                overlap_y = max(0, min(y + h, ry + rh) - max(y, ry))
                overlap_area = overlap_x * overlap_y
                min_area = min(w * h, rw * rh)
                if overlap_area > 0.5 * min_area:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            detected_regions.append((x, y, w, h))

            # Calculate confidence
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            circularity = (
                4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            )

            confidence = min(0.95, solidity * 0.5 + circularity * 0.2 + 0.3)

            if confidence < self.confidence_threshold:
                continue

            object_id += 1

            # Get rotated rectangle for accurate dimensions
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            rect_w, rect_h = rect[1]
            if rect_w < rect_h:
                rect_w, rect_h = rect_h, rect_w

            # Analyze for 3D detection
            roi = gray[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]

            is_3d = False
            depth_factor = 0.0

            if roi.size > 0:
                # Texture analysis
                texture_var = np.var(roi)

                # Gradient analysis
                sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobelx**2 + sobely**2)
                gradient_mean = np.mean(gradient_mag)

                # Edge density
                roi_edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(roi_edges > 0) / roi.size

                # Shadow detection (brightness gradient)
                h_third = roi.shape[0] // 3
                if h_third > 0:
                    top_bright = np.mean(roi[:h_third, :])
                    bottom_bright = np.mean(roi[-h_third:, :])
                    bright_diff = abs(top_bright - bottom_bright)
                else:
                    bright_diff = 0

                # Combine for 3D score
                texture_score = min(1.0, texture_var / 2000)
                gradient_score = min(1.0, gradient_mean / 50)
                edge_score = min(1.0, edge_density * 20)
                shadow_score = min(1.0, bright_diff / 50)

                depth_factor = (
                    texture_score * 0.2
                    + gradient_score * 0.3
                    + edge_score * 0.2
                    + shadow_score * 0.3
                )

                # 3D if significant depth or complex shape
                is_3d = depth_factor > 0.35 or len(approx) > 6 or solidity < 0.85

            detected_objects.append(
                {
                    "id": object_id,
                    "bbox": (x, y, w, h),
                    "rotated_rect": rect,
                    "box_points": box,
                    "rect_width_px": rect_w,
                    "rect_height_px": rect_h,
                    "center": (cx, cy),
                    "confidence": confidence,
                    "contour": contour,
                    "is_3d": is_3d,
                    "depth_factor": depth_factor,
                }
            )

        return detected_objects

    def _calculate_dimensions(
        self, obj: dict, pixels_per_cm: float
    ) -> Tuple[float, float, Optional[float]]:
        """
        Calculate real dimensions in cm
        """
        # Get pixel dimensions
        width_px = obj["rect_width_px"]
        height_px = obj["rect_height_px"]

        # Convert to cm
        length_cm = round(width_px / pixels_per_cm, 1)
        width_cm = round(height_px / pixels_per_cm, 1)

        # Ensure length >= width
        if length_cm < width_cm:
            length_cm, width_cm = width_cm, length_cm

        # Minimum 0.5 cm
        length_cm = max(0.5, length_cm)
        width_cm = max(0.5, width_cm)

        # Calculate height for 3D objects
        height_cm = None
        if obj["is_3d"]:
            depth_factor = obj["depth_factor"]

            # Estimate height based on depth factor
            # Typical objects: height is 20-80% of smallest dimension
            min_dim = min(length_cm, width_cm)
            height_cm = round(min_dim * (0.2 + depth_factor * 0.6), 1)
            height_cm = max(0.5, min(height_cm, min_dim * 1.5))

        return length_cm, width_cm, height_cm

    def _annotate_image(
        self,
        image: np.ndarray,
        objects: List[MeasuredObject3D],
        a4_corners: Optional[np.ndarray],
        a4_detected: bool,
        pixels_per_cm: float,
    ) -> np.ndarray:
        """Draw annotations on image"""
        annotated = image.copy()
        height, width = image.shape[:2]

        # Draw A4 paper outline
        if a4_corners is not None:
            pts = a4_corners.astype(np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)

            # Label
            cv2.putText(
                annotated,
                "A4 Paper (21 x 29.7 cm)",
                (int(pts[0][0]), int(pts[0][1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Calibration status
        if a4_detected:
            status = f"A4 Detected - {pixels_per_cm:.1f} px/cm"
            status_color = (0, 255, 0)
        else:
            status = "Place objects on A4 paper"
            status_color = (0, 165, 255)

        cv2.putText(
            annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
        )

        # Draw each object
        for obj in objects:
            x, y, w, h = obj.bounding_box

            # Color: green for 3D, blue for 2D
            if obj.object_type == ObjectType.OBJECT_3D:
                color = (0, 255, 0)
                bg_color = (0, 180, 0)
            else:
                color = (255, 165, 0)
                bg_color = (200, 130, 0)

            # Draw box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Corner markers
            m = min(20, w // 4, h // 4)
            t = 3
            cv2.line(annotated, (x, y), (x + m, y), color, t)
            cv2.line(annotated, (x, y), (x, y + m), color, t)
            cv2.line(annotated, (x + w, y), (x + w - m, y), color, t)
            cv2.line(annotated, (x + w, y), (x + w, y + m), color, t)
            cv2.line(annotated, (x, y + h), (x + m, y + h), color, t)
            cv2.line(annotated, (x, y + h), (x, y + h - m), color, t)
            cv2.line(annotated, (x + w, y + h), (x + w - m, y + h), color, t)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - m), color, t)

            # Labels
            type_label = f"{obj.object_type.value}: {obj.label}"
            if obj.object_type == ObjectType.OBJECT_3D and obj.height_cm:
                dim_text = f"L:{obj.length_cm} W:{obj.width_cm} H:{obj.height_cm} cm"
            else:
                dim_text = f"L:{obj.length_cm} W:{obj.width_cm} cm"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Type label at top
            (tw, th), _ = cv2.getTextSize(type_label, font, font_scale, thickness)
            cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 8, y), bg_color, -1)
            cv2.putText(
                annotated,
                type_label,
                (x + 4, y - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # Dimensions at bottom
            (dw, dh), _ = cv2.getTextSize(dim_text, font, font_scale, thickness)
            cv2.rectangle(
                annotated, (x, y + h), (x + dw + 8, y + h + dh + 8), bg_color, -1
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

            # Dimension arrows
            mid_y = y + h // 2
            mid_x = x + w // 2

            if w > 40:
                cv2.arrowedLine(
                    annotated,
                    (x + 5, mid_y),
                    (x + w - 5, mid_y),
                    (255, 255, 255),
                    2,
                    tipLength=0.05,
                )
            if h > 40:
                cv2.arrowedLine(
                    annotated,
                    (mid_x, y + 5),
                    (mid_x, y + h - 5),
                    (255, 255, 255),
                    2,
                    tipLength=0.05,
                )

        return annotated

    def process_frame(
        self, image: np.ndarray, return_annotated: bool = True
    ) -> RealtimeMeasurementResult:
        """
        Process frame and measure objects on A4 paper
        """
        start_time = time.time()
        height, width = image.shape[:2]

        try:
            # Step 1: Detect A4 paper
            a4_corners, pixels_per_cm = self._detect_a4_paper(image)
            a4_detected = a4_corners is not None

            if not a4_detected:
                pixels_per_cm = self._default_pixels_per_cm

            # Step 2: Detect objects on paper
            detected = self._detect_objects_on_paper(image, a4_corners, pixels_per_cm)

            if not detected:
                processing_time = (time.time() - start_time) * 1000

                # Still annotate to show A4 detection status
                annotated_base64 = None
                if return_annotated:
                    annotated = self._annotate_image(
                        image, [], a4_corners, a4_detected, pixels_per_cm
                    )
                    _, buffer = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    annotated_base64 = base64.b64encode(buffer).decode("utf-8")

                msg = "No objects detected."
                if not a4_detected:
                    msg += " Place objects on A4 paper for measurement."
                else:
                    msg += " Place objects on the A4 paper."

                return RealtimeMeasurementResult(
                    success=True,
                    message=msg,
                    objects=[],
                    frame_width=width,
                    frame_height=height,
                    processing_time_ms=round(processing_time, 2),
                    annotated_image_base64=annotated_base64,
                    calibration_info={
                        "reference_detected": a4_detected,
                        "reference_type": "a4_paper",
                        "pixels_per_cm": round(pixels_per_cm, 2),
                    },
                )

            # Step 3: Calculate dimensions
            measured_objects = []
            for obj in detected:
                length_cm, width_cm, height_cm = self._calculate_dimensions(
                    obj, pixels_per_cm
                )

                measured = MeasuredObject3D(
                    object_id=obj["id"],
                    object_type=ObjectType.OBJECT_3D
                    if obj["is_3d"]
                    else ObjectType.OBJECT_2D,
                    label=f"Object {obj['id']}",
                    confidence=round(obj["confidence"], 2),
                    length_cm=length_cm,
                    width_cm=width_cm,
                    height_cm=height_cm,
                    bounding_box=obj["bbox"],
                    center=obj["center"],
                )
                measured_objects.append(measured)

            # Step 4: Annotate
            annotated_base64 = None
            if return_annotated:
                annotated = self._annotate_image(
                    image, measured_objects, a4_corners, a4_detected, pixels_per_cm
                )
                _, buffer = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                annotated_base64 = base64.b64encode(buffer).decode("utf-8")

            processing_time = (time.time() - start_time) * 1000

            msg = f"Measured {len(measured_objects)} object(s)"
            if a4_detected:
                msg += " on A4 paper"
            else:
                msg += " (estimates - add A4 paper for accuracy)"

            return RealtimeMeasurementResult(
                success=True,
                message=msg,
                objects=measured_objects,
                frame_width=width,
                frame_height=height,
                processing_time_ms=round(processing_time, 2),
                annotated_image_base64=annotated_base64,
                calibration_info={
                    "reference_detected": a4_detected,
                    "reference_type": "a4_paper",
                    "pixels_per_cm": round(pixels_per_cm, 2),
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
        reference_type: str = "a4_paper",
        reference_width_cm: float = None,
        reference_height_cm: float = None,
    ):
        """Manual calibration (optional)"""
        # Adjust default pixels per cm based on distance
        self._default_pixels_per_cm = (
            40.0 * (30.0 / reference_distance_cm) * scale_factor
        )
        logger.info(
            f"Manual calibration: distance={reference_distance_cm}cm, px/cm={self._default_pixels_per_cm:.2f}"
        )

    def set_reference_type(self, ref_type, custom_width_cm=None, custom_height_cm=None):
        """For API compatibility - we only use A4 paper"""
        pass


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
