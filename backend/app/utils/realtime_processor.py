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
    MIN_OBJECT_AREA = 800  # Increased to ignore noise
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

    def _get_four_corners(self, hull: np.ndarray) -> Optional[np.ndarray]:
        """Convert a convex hull to exactly 4 corners using min area rect"""
        if len(hull) < 3:
            return None
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        return box

    def _detect_a4_paper(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect A4 paper in the image with improved robustness for dim/warm lighting
        Returns: (corner_points, pixels_per_cm)
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing for robust detection
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # CLAHE for contrast enhancement in dim lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        best_rect = None
        best_score = 0
        best_pixels_per_cm = self._default_pixels_per_cm

        # Try multiple thresholding strategies
        thresholds = [
            # Strategy A: Adaptive threshold
            cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5),
            # Strategy B: Otsu's threshold
            cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            # Strategy C: Fixed threshold fallback
            cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)[1]
        ]

        for white_mask in thresholds:
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(
                white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < (width * height * 0.08) or area > (width * height * 0.98):
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Relax to 4-8 points then take hull
                if 4 <= len(approx) <= 10:
                    hull = cv2.convexHull(approx)
                    rect_pts = self._get_four_corners(hull)
                    if rect_pts is not None:
                        ordered = self._order_points(rect_pts)

                        w1 = np.linalg.norm(ordered[1] - ordered[0])
                        w2 = np.linalg.norm(ordered[2] - ordered[3])
                        h1 = np.linalg.norm(ordered[3] - ordered[0])
                        h2 = np.linalg.norm(ordered[2] - ordered[1])

                        rect_width = (w1 + w2) / 2
                        rect_height = (h1 + h2) / 2

                        if rect_width < 80 or rect_height < 80:
                            continue

                        # Check A4 aspect ratio (1.414)
                        aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
                        if 1.1 < aspect < 1.8:
                            if rect_width > rect_height:
                                pixels_per_cm = rect_width / A4_HEIGHT_CM
                            else:
                                pixels_per_cm = rect_height / A4_HEIGHT_CM

                            # Score
                            solidity = area / cv2.contourArea(cv2.convexHull(contour)) if area > 0 else 0
                            ratio_score = 1.0 - min(abs(aspect - 1.414) / 1.414, 1.0)
                            size_score = min(area / (width * height * 0.4), 1.0)
                            score = solidity * 0.4 + size_score * 0.3 + ratio_score * 0.3

                            if score > best_score:
                                best_score = score
                                best_rect = ordered
                                best_pixels_per_cm = pixels_per_cm

        if best_score < 0.4:
            return None, self._default_pixels_per_cm
            
        return best_rect, best_pixels_per_cm

    def _detect_a4_edge_side_view(self, image: np.ndarray) -> Tuple[bool, float]:
        """Detect A4 paper edge in side view"""
        height, width = image.shape[:2]
        if self._top_view_pixels_per_cm > 0:
            return True, self._top_view_pixels_per_cm

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100, minLineLength=width * 0.3, maxLineGap=20
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:
                    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
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

        # Multi-strategy object detection
        
        # 1. Edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 80)
        edges = cv2.dilate(edges, kernel, iterations=2)
        if paper_mask is not None:
            edges = cv2.bitwise_and(edges, paper_mask)
        contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours1)

        # 2. Thresholding for dark objects
        _, dark_mask = cv2.threshold(enhanced, 130, 255, cv2.THRESH_BINARY_INV)
        if paper_mask is not None:
            dark_mask = cv2.bitwise_and(dark_mask, paper_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours2, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours2)

        # Sort and filter
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
        detected_regions = []
        objects = []
        object_id = 0

        for contour in all_contours[:30]:
            area = cv2.contourArea(contour)
            if area < self.MIN_OBJECT_AREA or area > self.MAX_OBJECT_AREA:
                continue

            # Check for hull-based area (prevents noise fragments)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area < self.MIN_OBJECT_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            # Check if inside paper
            if paper_mask is not None:
                if paper_mask[cy, cx] == 0:
                    continue
                # Skip if it's the paper itself
                if a4_corners is not None:
                    paper_area = cv2.contourArea(a4_corners.astype(np.int32).reshape(-1, 1, 2))
                    if area / paper_area > 0.8:
                        continue

            # Duplicate check (IOU)
            is_duplicate = False
            for rx, ry, rw, rh in detected_regions:
                overlap_x = max(0, min(x + w, rx + rw) - max(x, rx))
                overlap_y = max(0, min(y + h, ry + rh) - max(y, ry))
                if overlap_x * overlap_y > 0.5 * min(w * h, rw * rh):
                    is_duplicate = True
                    break
            
            if is_duplicate: continue
            detected_regions.append((x, y, w, h))

            object_id += 1
            rect = cv2.minAreaRect(contour)
            box_w, box_h = rect[1]
            if box_w < box_h: box_w, box_h = box_h, box_w

            length_cm = round(box_w / pixels_per_cm, 1)
            width_cm = round(box_h / pixels_per_cm, 1)
            height_cm = None
            obj_type = ObjectType.OBJECT_2D

            if view_type == ViewType.SIDE:
                height_cm = round(h / pixels_per_cm, 1)
                obj_type = ObjectType.OBJECT_3D
            else:
                # Top view 3D heuristic
                roi = gray[y : y + h, x : x + w]
                if roi.size > 0 and np.std(roi) > 30:
                    obj_type = ObjectType.OBJECT_3D
                    height_cm = round(min(length_cm, width_cm) * 0.4, 1)

            objects.append(
                MeasuredObject3D(
                    object_id=object_id,
                    object_type=obj_type,
                    label=f"Obj {object_id}",
                    confidence=0.9,
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

        if a4_corners is not None:
            pts = a4_corners.astype(np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
            cv2.putText(annotated, "A4 Paper", (int(pts[0][0]), int(pts[0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for obj in objects:
            x, y, w, h = obj.bounding_box
            color = (0, 255, 0) if obj.object_type == ObjectType.OBJECT_3D else (255, 255, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            if view_type == ViewType.TOP:
                label = f"L:{obj.length_cm} W:{obj.width_cm}cm"
            else:
                label = f"H:{obj.height_cm}cm"

            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        info = f"Scale: {pixels_per_cm:.1f} px/cm | {view_type.value.upper()}"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated

    def process_frame(
        self, image: np.ndarray, return_annotated: bool = True, view_type: str = "top"
    ) -> RealtimeMeasurementResult:
        """Process a camera frame and measure objects"""
        start_time = time.time()
        height, width = image.shape[:2]
        vtype = ViewType.TOP if view_type == "top" else ViewType.SIDE

        if vtype == ViewType.TOP:
            a4_corners, pixels_per_cm = self._detect_a4_paper(image)
            a4_detected = a4_corners is not None
            if a4_detected: self._top_view_pixels_per_cm = pixels_per_cm
            else: pixels_per_cm = self._default_pixels_per_cm

            paper_mask = None
            if a4_corners is not None:
                paper_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(paper_mask, [a4_corners.astype(np.int32)], 255)

            objects = self._detect_objects(image, paper_mask, pixels_per_cm, a4_corners, vtype)
            self._top_view_objects = objects
        else:
            a4_detected, pixels_per_cm = self._detect_a4_edge_side_view(image)
            a4_corners = None
            objects = self._detect_objects(image, None, pixels_per_cm, None, vtype)
            
            if self._top_view_objects and objects:
                for top_obj in self._top_view_objects:
                    for side_obj in objects:
                        top_obj.height_cm = side_obj.height_cm
                        top_obj.object_type = ObjectType.OBJECT_3D
                        break
                objects = self._top_view_objects

        annotated_b64 = None
        if return_annotated:
            annotated = self._draw_annotations(image, objects, a4_corners, pixels_per_cm, vtype)
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        proc_time = (time.time() - start_time) * 1000
        return RealtimeMeasurementResult(
            success=True,
            message=f"Detected {len(objects)} object(s)",
            objects=objects,
            frame_width=width,
            frame_height=height,
            processing_time_ms=round(proc_time, 2),
            annotated_image_base64=annotated_b64,
            calibration_info={
                "reference_detected": a4_detected,
                "reference_type": "a4_paper",
                "pixels_per_cm": float(pixels_per_cm),
            },
            view_type=vtype.value,
        )

    def calibrate(self, reference_distance_cm: float = 30.0, **kwargs):
        self._default_pixels_per_cm = 40.0 * (30.0 / reference_distance_cm)

    def reset(self):
        self._top_view_objects = []
        self._top_view_pixels_per_cm = 0


# Global processor instance
_processor: Optional[RealtimeProcessor] = None

def get_processor() -> RealtimeProcessor:
    global _processor
    if _processor is None:
        _processor = RealtimeProcessor()
    return _processor
