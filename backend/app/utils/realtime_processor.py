"""
Real-time Object Measurement Processor
Uses MiDaS for depth estimation and YOLOv8 for object detection
Measures 2D objects (length, breadth) and 3D objects (length, breadth, height)
"""

import cv2
import numpy as np
import base64
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import torch
import logging

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
    # Dimensions in centimeters
    length_cm: float
    breadth_cm: float
    height_cm: Optional[float]  # None for 2D objects
    # Bounding box [x, y, width, height]
    bounding_box: Tuple[int, int, int, int]
    # Center point
    center: Tuple[int, int]
    # Depth value (average) in relative units
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
    Real-time processor for object measurement using depth estimation
    """

    # Default camera parameters (can be calibrated)
    # These are approximate values for typical smartphone cameras
    DEFAULT_FOCAL_LENGTH_MM = 4.0  # Typical smartphone focal length
    DEFAULT_SENSOR_WIDTH_MM = 6.0  # Typical smartphone sensor width
    DEFAULT_REFERENCE_DISTANCE_CM = 100.0  # Assumed distance for calibration

    # Depth variance threshold to distinguish 2D from 3D
    DEPTH_VARIANCE_THRESHOLD = 0.15

    # Minimum object size (pixels) to consider
    MIN_OBJECT_SIZE = 50

    def __init__(
        self,
        use_gpu: bool = False,
        model_type: str = "midas_small",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the real-time processor

        Args:
            use_gpu: Whether to use GPU acceleration
            model_type: MiDaS model type ("midas_small", "dpt_hybrid", "dpt_large")
            confidence_threshold: Minimum confidence for object detection
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type

        # Models will be loaded lazily
        self._depth_model = None
        self._depth_transform = None
        self._object_detector = None

        # Calibration parameters
        self.focal_length_mm = self.DEFAULT_FOCAL_LENGTH_MM
        self.sensor_width_mm = self.DEFAULT_SENSOR_WIDTH_MM
        self.reference_distance_cm = self.DEFAULT_REFERENCE_DISTANCE_CM

        # Cache for depth scale factor
        self._depth_scale_factor = 1.0

        logger.info(
            f"RealtimeProcessor initialized (GPU: {self.use_gpu}, Model: {model_type})"
        )

    def _load_depth_model(self):
        """Load MiDaS depth estimation model"""
        if self._depth_model is not None:
            return

        logger.info(f"Loading MiDaS depth model: {self.model_type}")

        try:
            # Load MiDaS model
            if self.model_type == "midas_small":
                self._depth_model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
                )
                self._depth_transform = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True
                ).small_transform
            elif self.model_type == "dpt_hybrid":
                self._depth_model = torch.hub.load(
                    "intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True
                )
                self._depth_transform = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True
                ).dpt_transform
            else:
                # Default to small for speed
                self._depth_model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
                )
                self._depth_transform = torch.hub.load(
                    "intel-isl/MiDaS", "transforms", trust_repo=True
                ).small_transform

            self._depth_model.to(self.device)
            self._depth_model.eval()

            logger.info("MiDaS depth model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise RuntimeError(f"Could not load depth model: {e}")

    def _load_object_detector(self):
        """Load YOLOv8 object detection model"""
        if self._object_detector is not None:
            return

        logger.info("Loading YOLOv8 object detector")

        try:
            from ultralytics import YOLO

            # Use YOLOv8 nano for speed, can use 's' or 'm' for better accuracy
            self._object_detector = YOLO("yolov8n.pt")

            logger.info("YOLOv8 object detector loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLOv8: {e}")
            raise RuntimeError(f"Could not load object detector: {e}")

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image using MiDaS

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Depth map (higher values = closer)
        """
        self._load_depth_model()

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform
        input_batch = self._depth_transform(img_rgb).to(self.device)

        # Predict depth
        with torch.no_grad():
            depth_prediction = self._depth_model(input_batch)

            # Interpolate to original size
            depth_prediction = torch.nn.functional.interpolate(
                depth_prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = depth_prediction.cpu().numpy()

        # Normalize depth map (0-1, where 1 is closer)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map

    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in image using YOLOv8

        Args:
            image: BGR image

        Returns:
            List of detected objects with bounding boxes
        """
        self._load_object_detector()

        # Run detection
        results = self._object_detector(image, verbose=False)

        detected_objects = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                # Calculate dimensions
                width = x2 - x1
                height = y2 - y1

                # Filter out too small objects
                if width < self.MIN_OBJECT_SIZE or height < self.MIN_OBJECT_SIZE:
                    continue

                detected_objects.append(
                    {
                        "bbox": (x1, y1, width, height),
                        "xyxy": (x1, y1, x2, y2),
                        "label": cls_name,
                        "confidence": conf,
                        "center": (x1 + width // 2, y1 + height // 2),
                    }
                )

        return detected_objects

    def _calculate_real_dimensions(
        self,
        bbox: Tuple[int, int, int, int],
        depth_map: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, Optional[float], ObjectType, float]:
        """
        Calculate real-world dimensions from bounding box and depth

        Args:
            bbox: (x, y, width, height) bounding box
            depth_map: Normalized depth map
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            (length_cm, breadth_cm, height_cm, object_type, avg_depth)
        """
        x, y, w, h = bbox

        # Ensure bounds are within image
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_width, x + w), min(image_height, y + h)

        # Extract depth region for the object
        depth_region = depth_map[y1:y2, x1:x2]

        if depth_region.size == 0:
            return 0.0, 0.0, None, ObjectType.OBJECT_2D, 0.0

        # Calculate depth statistics
        avg_depth = float(np.mean(depth_region))
        depth_variance = float(np.var(depth_region))
        depth_std = float(np.std(depth_region))

        # Determine if object is 2D or 3D based on depth variance
        is_3d = depth_std > self.DEPTH_VARIANCE_THRESHOLD

        # Calculate distance factor (inverse of normalized depth)
        # MiDaS output: higher = closer, so we invert
        distance_factor = 1.0 / (
            avg_depth + 0.1
        )  # Add small value to avoid division by zero

        # Calculate pixel-to-cm conversion
        # Using pinhole camera model approximation
        focal_length_pixels = (
            self.focal_length_mm / self.sensor_width_mm
        ) * image_width

        # Estimate real-world size
        # At reference_distance, we calibrate the conversion
        scale = (self.reference_distance_cm * distance_factor) / focal_length_pixels

        # Apply calibration scale factor
        scale *= self._depth_scale_factor

        # Length and Breadth (in the image plane)
        length_cm = round(w * scale, 1)
        breadth_cm = round(h * scale, 1)

        # Height (depth dimension) for 3D objects
        height_cm = None
        if is_3d:
            # Estimate height from depth variation
            depth_range = float(np.max(depth_region) - np.min(depth_region))
            # Convert depth range to real-world height
            height_cm = round(depth_range * self.reference_distance_cm * 0.5, 1)
            # Ensure minimum reasonable height
            if height_cm < 1.0:
                height_cm = round(min(length_cm, breadth_cm) * 0.3, 1)

        object_type = ObjectType.OBJECT_3D if is_3d else ObjectType.OBJECT_2D

        return length_cm, breadth_cm, height_cm, object_type, avg_depth

    def _annotate_image(
        self, image: np.ndarray, objects: List[MeasuredObject3D]
    ) -> np.ndarray:
        """
        Draw measurement annotations on image

        Args:
            image: Original BGR image
            objects: List of measured objects

        Returns:
            Annotated image
        """
        annotated = image.copy()

        for obj in objects:
            x, y, w, h = obj.bounding_box
            center_x, center_y = obj.center

            # Color based on object type
            if obj.object_type == ObjectType.OBJECT_3D:
                color = (0, 255, 0)  # Green for 3D
                box_color = (0, 200, 0)
            else:
                color = (255, 165, 0)  # Orange for 2D
                box_color = (200, 130, 0)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # Draw corner markers
            marker_size = 10
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            for cx, cy in corners:
                cv2.line(
                    annotated, (cx - marker_size, cy), (cx + marker_size, cy), color, 2
                )
                cv2.line(
                    annotated, (cx, cy - marker_size), (cx, cy + marker_size), color, 2
                )

            # Prepare measurement text
            type_text = f"[{obj.object_type.value}] {obj.label}"

            if obj.object_type == ObjectType.OBJECT_3D:
                dim_text = (
                    f"L:{obj.length_cm}cm B:{obj.breadth_cm}cm H:{obj.height_cm}cm"
                )
            else:
                dim_text = f"L:{obj.length_cm}cm B:{obj.breadth_cm}cm"

            # Background for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Type label at top
            (text_w, text_h), _ = cv2.getTextSize(
                type_text, font, font_scale, thickness
            )
            cv2.rectangle(
                annotated, (x, y - text_h - 10), (x + text_w + 10, y), color, -1
            )
            cv2.putText(
                annotated,
                type_text,
                (x + 5, y - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

            # Dimensions at bottom
            (dim_w, dim_h), _ = cv2.getTextSize(dim_text, font, font_scale, thickness)
            cv2.rectangle(
                annotated, (x, y + h), (x + dim_w + 10, y + h + dim_h + 10), color, -1
            )
            cv2.putText(
                annotated,
                dim_text,
                (x + 5, y + h + dim_h + 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

            # Draw dimension arrows
            # Horizontal (Length)
            mid_y = y + h // 2
            cv2.arrowedLine(
                annotated, (x, mid_y), (x + w, mid_y), color, 2, tipLength=0.05
            )

            # Vertical (Breadth)
            mid_x = x + w // 2
            cv2.arrowedLine(
                annotated, (mid_x, y), (mid_x, y + h), color, 2, tipLength=0.05
            )

        return annotated

    def process_frame(
        self,
        image: np.ndarray,
        return_annotated: bool = True,
        reference_object_cm: Optional[Tuple[float, float]] = None,
    ) -> RealtimeMeasurementResult:
        """
        Process a single frame and measure objects

        Args:
            image: BGR image (OpenCV format)
            return_annotated: Whether to return annotated image
            reference_object_cm: If provided, use first detected object as reference
                                with these dimensions (width_cm, height_cm) for calibration

        Returns:
            RealtimeMeasurementResult with measurements
        """
        import time

        start_time = time.time()

        height, width = image.shape[:2]

        try:
            # Step 1: Detect objects
            detected_objects = self._detect_objects(image)

            if not detected_objects:
                return RealtimeMeasurementResult(
                    success=True,
                    message="No objects detected in frame",
                    objects=[],
                    frame_width=width,
                    frame_height=height,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            # Step 2: Estimate depth
            depth_map = self._estimate_depth(image)

            # Step 3: Calibrate if reference provided
            if reference_object_cm and len(detected_objects) > 0:
                ref_obj = detected_objects[0]
                ref_bbox = ref_obj["bbox"]
                ref_width_px = ref_bbox[2]
                ref_height_px = ref_bbox[3]

                # Calculate calibration factor
                expected_width_cm, expected_height_cm = reference_object_cm

                # Use average of width and height for calibration
                actual_width_cm, actual_height_cm, _, _, _ = (
                    self._calculate_real_dimensions(ref_bbox, depth_map, width, height)
                )

                if actual_width_cm > 0 and actual_height_cm > 0:
                    width_factor = expected_width_cm / actual_width_cm
                    height_factor = expected_height_cm / actual_height_cm
                    self._depth_scale_factor = (width_factor + height_factor) / 2

            # Step 4: Calculate dimensions for each object
            measured_objects = []

            for idx, obj in enumerate(detected_objects):
                bbox = obj["bbox"]

                length_cm, breadth_cm, height_cm, obj_type, depth_val = (
                    self._calculate_real_dimensions(bbox, depth_map, width, height)
                )

                measured_obj = MeasuredObject3D(
                    object_id=idx + 1,
                    object_type=obj_type,
                    label=obj["label"],
                    confidence=obj["confidence"],
                    length_cm=length_cm,
                    breadth_cm=breadth_cm,
                    height_cm=height_cm,
                    bounding_box=bbox,
                    center=obj["center"],
                    depth_value=depth_val,
                )

                measured_objects.append(measured_obj)

            # Step 5: Annotate image if requested
            annotated_base64 = None
            if return_annotated:
                annotated_img = self._annotate_image(image, measured_objects)
                _, buffer = cv2.imencode(
                    ".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                annotated_base64 = base64.b64encode(buffer).decode("utf-8")

            processing_time = (time.time() - start_time) * 1000

            return RealtimeMeasurementResult(
                success=True,
                message=f"Measured {len(measured_objects)} object(s)",
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
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def calibrate(
        self,
        reference_distance_cm: float = 100.0,
        focal_length_mm: float = 4.0,
        sensor_width_mm: float = 6.0,
    ):
        """
        Set camera calibration parameters

        Args:
            reference_distance_cm: Expected distance from camera to objects
            focal_length_mm: Camera focal length in mm
            sensor_width_mm: Camera sensor width in mm
        """
        self.reference_distance_cm = reference_distance_cm
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm

        logger.info(
            f"Calibration set: distance={reference_distance_cm}cm, "
            f"focal={focal_length_mm}mm, sensor={sensor_width_mm}mm"
        )


# Global processor instance (lazy loaded)
_processor: Optional[RealtimeProcessor] = None


def get_processor() -> RealtimeProcessor:
    """Get or create the global processor instance"""
    global _processor
    if _processor is None:
        _processor = RealtimeProcessor(
            use_gpu=False,  # CPU for Render deployment
            model_type="midas_small",  # Fastest model
            confidence_threshold=0.4,
        )
    return _processor


def process_frame_for_measurement(
    image_data: bytes, return_annotated: bool = True
) -> RealtimeMeasurementResult:
    """
    Convenience function to process image bytes

    Args:
        image_data: Raw image bytes (JPEG/PNG)
        return_annotated: Whether to return annotated image

    Returns:
        Measurement result
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
