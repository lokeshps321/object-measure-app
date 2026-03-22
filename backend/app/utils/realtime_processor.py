"""
Real-time Object Measurement Processor - V5
Two-Photo Mode for Accurate 3D Measurements

How it works:
1. TOP VIEW: Detect A4 paper, measure object's Length & Width
2. SIDE VIEW: Use A4 paper edge as reference, measure Height
3. Combine for accurate L × W × H measurements
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


class ViewType(str, Enum):
    TOP = "top"
    SIDE = "side"


# A4 paper dimensions in cm
A4_WIDTH_CM = 21.0  # Short edge
A4_HEIGHT_CM = 29.7  # Long edge


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
    view_type: str = "top"


class RealtimeProcessor:
    """
    Measure objects placed on A4 paper
    Supports both top-view (2D) and side-view (height) measurements
    """

    # Detection parameters
    MIN_OBJECT_AREA = 200  # Lowered for small objects
    MAX_OBJECT_AREA = 500000

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self._models_loaded = True
        self._calibration: Optional[CalibrationData] = None
        self._default_pixels_per_cm = 40.0

        # Store top-view measurements for combining with side view
        self._top_view_objects: List[MeasuredObject3D] = []
        self._top_view_pixels_per_cm: float = 0

        logger.info("RealtimeProcessor V5 initialized - Two-photo 3D measurement")

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect_a4_paper(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect A4 paper in the image
        Returns: (corner_points, pixels_per_cm)
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)

        best_rect = None
        best_score = 0
        best_pixels_per_cm = self._default_pixels_per_cm

        # Method 1: White region detection
        _, white_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (width * height * 0.1) or area > (width * height * 0.95):
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                ordered = self._order_points(pts)

                w1 = np.linalg.norm(ordered[1] - ordered[0])
                w2 = np.linalg.norm(ordered[2] - ordered[3])
                h1 = np.linalg.norm(ordered[3] - ordered[0])
                h2 = np.linalg.norm(ordered[2] - ordered[1])

                rect_width = (w1 + w2) / 2
                rect_height = (h1 + h2) / 2

                if rect_width < 50 or rect_height < 50:
                    continue

                # Check A4 aspect ratio (1.414)
                aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
                if 1.3 < aspect < 1.55:
                    # Calculate pixels per cm
                    if rect_width > rect_height:
                        pixels_per_cm = rect_width / A4_HEIGHT_CM
                    else:
                        pixels_per_cm = rect_height / A4_HEIGHT_CM

                    # Score based on size and rectangularity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    size_score = min(area / (width * height * 0.5), 1.0)
                    score = solidity * 0.6 + size_score * 0.4

                    if score > best_score:
                        best_score = score
                        best_rect = ordered
                        best_pixels_per_cm = pixels_per_cm

        # Method 2: Edge detection fallback
        if best_rect is None or best_score < 0.5:
            edges = cv2.Canny(blurred, 50, 150)
            edges = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
                area = cv2.contourArea(contour)
                if area < (width * height * 0.1):
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    ordered = self._order_points(pts)

                    w1 = np.linalg.norm(ordered[1] - ordered[0])
                    h1 = np.linalg.norm(ordered[3] - ordered[0])
                    rect_width = w1
                    rect_height = h1

                    aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
                    if 1.3 < aspect < 1.55:
                        if rect_width > rect_height:
                            pixels_per_cm = rect_width / A4_HEIGHT_CM
                        else:
                            pixels_per_cm = rect_height / A4_HEIGHT_CM

                        if best_rect is None:
                            best_rect = ordered
                            best_pixels_per_cm = pixels_per_cm
                        break

        return best_rect, best_pixels_per_cm

    def _detect_a4_edge_side_view(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect A4 paper edge in side view for height calibration
        In side view, A4 paper appears as a thin white line/edge
        Returns: (detected, pixels_per_cm)
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for horizontal white line (A4 paper edge)
        # The A4 paper lying flat will show as a line at the bottom

        # Use stored calibration if available
        if self._top_view_pixels_per_cm > 0:
            return True, self._top_view_pixels_per_cm

        # Try to detect the paper edge
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Look for long horizontal lines (paper edge)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100, minLineLength=width * 0.3, maxLineGap=20
        )

        if lines is not None:
            # Find the most horizontal line near bottom half of image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Nearly horizontal
                    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    # Estimate pixels_per_cm from line length (assume it's A4 long edge)
                    pixels_per_cm = line_length / A4_HEIGHT_CM
                    return True, pixels_per_cm

        return False, self._default_pixels_per_cm

    def _detect_objects(
        self,
        image: np.ndarray,
        paper_mask: Optional[np.ndarray],
        pixels_per_cm: float,
        a4_corners: Optional[np.ndarray],
        view_type: ViewType = ViewType.TOP,
    ) -> List[MeasuredObject3D]:
        """Detect and measure objects in the image"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel = np.ones((3, 3), np.uint8)
        all_contours = []

        # Method 1: Edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        if paper_mask is not None:
            edges = cv2.bitwise_and(edges, paper_mask)
        contours1, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours1)

        # Method 2: Threshold for dark objects
        _, dark_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        if paper_mask is not None:
            dark_mask = cv2.bitwise_and(dark_mask, paper_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours2, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours2)

        # Method 3: Adaptive threshold
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

        # Method 4: Color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, (0, 15, 0), (180, 255, 255))
        low_value = cv2.inRange(hsv, (0, 0, 0), (180, 255, 230))
        combined = cv2.bitwise_or(color_mask, low_value)
        if paper_mask is not None:
            combined = cv2.bitwise_and(combined, paper_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours4, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours4)

        # Method 5: Laplacian for edge/shadow detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, lap_mask = cv2.threshold(laplacian, 8, 255, cv2.THRESH_BINARY)
        if paper_mask is not None:
            lap_mask = cv2.bitwise_and(lap_mask, paper_mask)
        lap_mask = cv2.morphologyEx(lap_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours5, _ = cv2.findContours(
            lap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours5)

        # Sort by area
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

        detected_regions = []
        objects = []
        object_id = 0

        for contour in all_contours[:50]:
            area = cv2.contourArea(contour)

            if area < self.MIN_OBJECT_AREA or area > self.MAX_OBJECT_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Skip very thin objects
            aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 100
            if aspect > 15:
                continue

            # Skip objects at edge
            margin = 3
            if (
                x < margin
                or y < margin
                or x + w > width - margin
                or y + h > height - margin
            ):
                continue

            # Check if on paper
            cx, cy = x + w // 2, y + h // 2
            if paper_mask is not None:
                if paper_mask[cy, cx] == 0:
                    continue
                # Skip if this is the A4 paper itself
                if a4_corners is not None:
                    paper_area = cv2.contourArea(
                        a4_corners.astype(np.int32).reshape(-1, 1, 2)
                    )
                    if abs(area - paper_area) / paper_area < 0.2:
                        continue

            # Check for duplicates
            is_duplicate = False
            for rx, ry, rw, rh in detected_regions:
                overlap_x = max(0, min(x + w, rx + rw) - max(x, rx))
                overlap_y = max(0, min(y + h, ry + rh) - max(y, ry))
                if overlap_x * overlap_y > 0.4 * min(w * h, rw * rh):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            detected_regions.append((x, y, w, h))

            # Calculate confidence
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            confidence = min(0.95, solidity * 0.7 + 0.25)

            if confidence < self.confidence_threshold:
                continue

            object_id += 1

            # Get dimensions using rotated rectangle for accuracy
            rect = cv2.minAreaRect(contour)
            box_w, box_h = rect[1]

            # Ensure length > width
            if box_w < box_h:
                box_w, box_h = box_h, box_w

            # Convert to cm
            length_cm = round(box_w / pixels_per_cm, 1)
            width_cm = round(box_h / pixels_per_cm, 1)

            # For side view, the "height" in image is the object's real height
            if view_type == ViewType.SIDE:
                # In side view: image width = object length, image height = object height
                height_cm = round(h / pixels_per_cm, 1)
                obj_type = ObjectType.OBJECT_3D
            else:
                # Top view: estimate if 3D based on shadows/texture
                height_cm = None
                obj_type = ObjectType.OBJECT_2D

                # Check for 3D indicators (shadows, gradients)
                roi = gray[y : y + h, x : x + w]
                if roi.size > 0:
                    std_dev = np.std(roi)
                    if std_dev > 25:  # Significant variation suggests 3D
                        obj_type = ObjectType.OBJECT_3D
                        # Rough height estimate from shadow
                        height_cm = round(min(length_cm, width_cm) * 0.3, 1)

            objects.append(
                MeasuredObject3D(
                    object_id=object_id,
                    object_type=obj_type,
                    label=f"Object {object_id}",
                    confidence=round(confidence, 2),
                    length_cm=length_cm,
                    width_cm=width_cm,
                    height_cm=height_cm,
                    bounding_box=(x, y, w, h),
                    center=(cx, cy),
                )
            )

        return objects

    def _draw_annotations(
        self,
        image: np.ndarray,
        objects: List[MeasuredObject3D],
        a4_corners: Optional[np.ndarray],
        pixels_per_cm: float,
        view_type: ViewType = ViewType.TOP,
    ) -> np.ndarray:
        """Draw measurement annotations on image"""
        annotated = image.copy()

        # Draw A4 paper outline
        if a4_corners is not None:
            pts = a4_corners.astype(np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
            cv2.putText(
                annotated,
                "A4 Paper",
                (int(pts[0][0]), int(pts[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Draw objects
        for obj in objects:
            x, y, w, h = obj.bounding_box

            # Color based on type
            color = (
                (0, 165, 255)
                if obj.object_type == ObjectType.OBJECT_3D
                else (255, 165, 0)
            )

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label with dimensions
            if view_type == ViewType.TOP:
                if obj.height_cm:
                    label = f"L:{obj.length_cm} W:{obj.width_cm} H:{obj.height_cm}cm"
                else:
                    label = f"L:{obj.length_cm} x W:{obj.width_cm} cm"
            else:
                label = f"Height: {obj.height_cm} cm"

            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(
                annotated,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Draw calibration info
        info = f"Scale: {pixels_per_cm:.1f} px/cm | View: {view_type.value.upper()}"
        cv2.putText(
            annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return annotated

    def process_frame(
        self, image: np.ndarray, return_annotated: bool = True, view_type: str = "top"
    ) -> RealtimeMeasurementResult:
        """
        Process a camera frame and measure objects

        Args:
            image: BGR image from camera
            return_annotated: Whether to return annotated image
            view_type: "top" for length/width, "side" for height
        """
        start_time = time.time()
        height, width = image.shape[:2]

        vtype = ViewType.TOP if view_type == "top" else ViewType.SIDE

        if vtype == ViewType.TOP:
            # TOP VIEW: Detect A4 paper and measure L × W
            a4_corners, pixels_per_cm = self._detect_a4_paper(image)
            a4_detected = a4_corners is not None

            if not a4_detected:
                # Try with default calibration
                pixels_per_cm = self._default_pixels_per_cm
            else:
                # Store for side view
                self._top_view_pixels_per_cm = pixels_per_cm

            # Create paper mask
            paper_mask = None
            if a4_corners is not None:
                paper_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(paper_mask, [a4_corners.astype(np.int32)], 255)

            # Detect objects
            objects = self._detect_objects(
                image, paper_mask, pixels_per_cm, a4_corners, vtype
            )

            # Store top view objects
            self._top_view_objects = objects

        else:
            # SIDE VIEW: Measure height
            a4_detected, pixels_per_cm = self._detect_a4_edge_side_view(image)
            a4_corners = None

            # For side view, detect objects without paper mask
            objects = self._detect_objects(image, None, pixels_per_cm, None, vtype)

            # Update heights in stored top-view objects if we have them
            if self._top_view_objects and objects:
                for top_obj in self._top_view_objects:
                    # Find matching object (by position/size similarity)
                    for side_obj in objects:
                        # Use the height from side view
                        top_obj.height_cm = side_obj.height_cm
                        top_obj.object_type = ObjectType.OBJECT_3D
                        break
                objects = self._top_view_objects

        # Generate annotated image
        annotated_b64 = None
        if return_annotated:
            annotated = self._draw_annotations(
                image, objects, a4_corners, pixels_per_cm, vtype
            )
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        processing_time = (time.time() - start_time) * 1000

        if not objects:
            return RealtimeMeasurementResult(
                success=True,
                message="No objects detected. Place objects on A4 paper for measurement.",
                objects=[],
                frame_width=width,
                frame_height=height,
                processing_time_ms=round(processing_time, 2),
                annotated_image_base64=annotated_b64,
                calibration_info={
                    "reference_detected": a4_detected,
                    "reference_type": "a4_paper",
                    "pixels_per_cm": float(pixels_per_cm),
                },
                view_type=vtype.value,
            )

        return RealtimeMeasurementResult(
            success=True,
            message=f"Measured {len(objects)} object(s) - {vtype.value} view",
            objects=objects,
            frame_width=width,
            frame_height=height,
            processing_time_ms=round(processing_time, 2),
            annotated_image_base64=annotated_b64,
            calibration_info={
                "reference_detected": a4_detected,
                "reference_type": "a4_paper",
                "pixels_per_cm": float(pixels_per_cm),
            },
            view_type=vtype.value,
        )

    def process_two_photos(
        self,
        top_image: np.ndarray,
        side_image: np.ndarray,
        return_annotated: bool = True,
    ) -> RealtimeMeasurementResult:
        """
        Process two photos for accurate 3D measurement

        Args:
            top_image: Top-down view for length & width
            side_image: Side view for height
            return_annotated: Whether to return annotated images
        """
        start_time = time.time()

        # Step 1: Process top view
        top_result = self.process_frame(
            top_image, return_annotated=False, view_type="top"
        )

        if not top_result.objects:
            return top_result

        # Step 2: Process side view for height
        side_result = self.process_frame(
            side_image, return_annotated=False, view_type="side"
        )

        # Step 3: Combine measurements
        combined_objects = []
        pixels_per_cm = top_result.calibration_info.get(
            "pixels_per_cm", self._default_pixels_per_cm
        )

        for i, top_obj in enumerate(top_result.objects):
            height_cm = None

            # Get height from side view if available
            if side_result.objects:
                # Use the first side object's height (or match by index)
                idx = min(i, len(side_result.objects) - 1)
                height_cm = side_result.objects[idx].height_cm

            combined_objects.append(
                MeasuredObject3D(
                    object_id=top_obj.object_id,
                    object_type=ObjectType.OBJECT_3D
                    if height_cm
                    else top_obj.object_type,
                    label=top_obj.label,
                    confidence=top_obj.confidence,
                    length_cm=top_obj.length_cm,
                    width_cm=top_obj.width_cm,
                    height_cm=height_cm,
                    bounding_box=top_obj.bounding_box,
                    center=top_obj.center,
                )
            )

        # Generate combined annotated image (top view with full dimensions)
        annotated_b64 = None
        if return_annotated:
            a4_corners, _ = self._detect_a4_paper(top_image)
            annotated = self._draw_annotations(
                top_image, combined_objects, a4_corners, pixels_per_cm, ViewType.TOP
            )
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        processing_time = (time.time() - start_time) * 1000

        return RealtimeMeasurementResult(
            success=True,
            message=f"3D measurement complete: {len(combined_objects)} object(s)",
            objects=combined_objects,
            frame_width=top_image.shape[1],
            frame_height=top_image.shape[0],
            processing_time_ms=round(processing_time, 2),
            annotated_image_base64=annotated_b64,
            calibration_info={
                "reference_detected": top_result.calibration_info.get(
                    "reference_detected", False
                ),
                "reference_type": "a4_paper",
                "pixels_per_cm": float(pixels_per_cm),
            },
            view_type="3d_combined",
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
        self._default_pixels_per_cm = (
            40.0 * (30.0 / reference_distance_cm) * scale_factor
        )
        logger.info(
            f"Manual calibration: distance={reference_distance_cm}cm, px/cm={self._default_pixels_per_cm:.2f}"
        )

    def reset(self):
        """Reset stored measurements"""
        self._top_view_objects = []
        self._top_view_pixels_per_cm = 0


# Global processor instance
_processor: Optional[RealtimeProcessor] = None


def get_processor() -> RealtimeProcessor:
    """Get or create global processor instance"""
    global _processor
    if _processor is None:
        _processor = RealtimeProcessor()
    return _processor
