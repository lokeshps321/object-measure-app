"""
Image Processing Utilities for Object Size Measurement
Uses HuggingFace Space AI model for depth estimation + measurement
Falls back to local OpenCV processing if HF Space is unavailable
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import base64
import httpx
import json
import logging
import time

logger = logging.getLogger(__name__)

# HuggingFace Space API URL
HF_SPACE_URL = "https://loke007-object-measure-ai.hf.space"

@dataclass
class MeasuredObject:
    """Represents a measured object with its dimensions"""
    object_id: int
    object_type: str  # "2D" or "3D"
    label: str
    confidence: float
    length_cm: float
    width_cm: float
    height_cm: Optional[float]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]


@dataclass
class MeasurementResult:
    """Complete measurement result"""
    success: bool
    message: str
    objects: List[MeasuredObject]
    reference_detected: bool
    processed_image_base64: Optional[str] = None
    processed_side_image_base64: Optional[str] = None
    mode: str = "2d"
    calibration_info: Optional[dict] = None


def measure_objects_via_hf(
    image_base64: str,
    mode: str = "2d",
    camera_distance_cm: float = 30.0,
) -> MeasurementResult:
    """
    Measure objects by calling the HuggingFace Space API.
    """
    try:
        logger.info(f"Calling HF Space for measurement (mode={mode}, dist={camera_distance_cm}cm)")
        start_time = time.time()

        # Call Gradio API
        # Gradio API endpoint format
        api_url = f"{HF_SPACE_URL}/api/predict"
        
        # Prepare the image as data URL
        if not image_base64.startswith("data:"):
            img_data_url = f"data:image/jpeg;base64,{image_base64}"
        else:
            img_data_url = image_base64

        payload = {
            "data": [
                img_data_url,  # image
                mode,          # mode
                camera_distance_cm,  # distance
            ]
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(api_url, json=payload)
            response.raise_for_status()

        result = response.json()
        elapsed = time.time() - start_time
        logger.info(f"HF Space responded in {elapsed:.1f}s")

        # Parse result
        data = result.get("data", [])
        if len(data) < 2:
            return MeasurementResult(
                success=False,
                message="Invalid response from AI model",
                objects=[],
                reference_detected=False,
            )

        # data[0] = annotated image (base64 data URL or path)
        # data[1] = JSON string with measurements
        annotated_img_data = data[0]
        measurements_json = json.loads(data[1])

        # Extract annotated image base64
        processed_image = None
        if annotated_img_data:
            if isinstance(annotated_img_data, dict) and "url" in annotated_img_data:
                # Gradio returns a URL - download it
                img_url = annotated_img_data["url"]
                if not img_url.startswith("http"):
                    img_url = f"{HF_SPACE_URL}{img_url}"
                with httpx.Client(timeout=30.0) as client:
                    img_resp = client.get(img_url)
                    processed_image = base64.b64encode(img_resp.content).decode("utf-8")
            elif isinstance(annotated_img_data, str) and annotated_img_data.startswith("data:"):
                processed_image = annotated_img_data.split(",", 1)[1]

        # Build measured objects
        measured_objects = []
        for obj in measurements_json.get("objects", []):
            measured_objects.append(
                MeasuredObject(
                    object_id=obj.get("id", 0),
                    object_type=obj.get("object_type", "2D"),
                    label=obj.get("label", "Object"),
                    confidence=obj.get("confidence", 0.5),
                    length_cm=obj.get("length_cm", 0),
                    width_cm=obj.get("width_cm", 0),
                    height_cm=obj.get("height_cm"),
                    bounding_box=tuple(obj.get("bbox", (0, 0, 0, 0))),
                    center=tuple(obj.get("center", (0, 0))),
                )
            )

        return MeasurementResult(
            success=measurements_json.get("success", True),
            message=measurements_json.get("message", f"Measured {len(measured_objects)} object(s)"),
            objects=measured_objects,
            reference_detected=True,
            processed_image_base64=processed_image,
            mode=mode,
            calibration_info={
                "method": "ai_depth_estimation",
                "camera_distance_cm": camera_distance_cm,
                "model": "Depth-Anything-V2",
            },
        )

    except httpx.TimeoutException:
        logger.warning("HF Space timeout, falling back to local processing")
        return None  # Signal to use fallback
    except Exception as e:
        logger.error(f"HF Space error: {e}")
        return None  # Signal to use fallback


def measure_objects_local(
    image: np.ndarray,
    mode: str = "2d",
    camera_distance_cm: float = 30.0,
    side_image: Optional[np.ndarray] = None,
    side_camera_distance_cm: float = 30.0,
) -> MeasurementResult:
    """
    Local fallback: measure objects using only OpenCV (no AI model).
    Uses edge detection + pinhole camera model.
    """
    h, w = image.shape[:2]
    
    # Camera model
    focal_px = w * 0.70
    cm_per_px = camera_distance_cm / focal_px

    # Calculate actual Height from Side View if provided
    actual_height_cm = None
    processed_side_image = None
    if mode == "3d" and side_image is not None:
        sh, sw = side_image.shape[:2]
        sfocal_px = sw * 0.70
        scm_per_px = side_camera_distance_cm / sfocal_px

        sgray = cv2.cvtColor(side_image, cv2.COLOR_BGR2GRAY)
        sclahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        senhanced = sclahe.apply(sgray)
        sblurred = cv2.GaussianBlur(senhanced, (5, 5), 1)

        skernel = np.ones((5, 5), np.uint8)
        sall_contours = []

        sedges = cv2.Canny(sblurred, 40, 120)
        sedges = cv2.dilate(sedges, skernel, iterations=2)
        sc1, _ = cv2.findContours(sedges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sall_contours.extend(sc1)

        _, sotsu = cv2.threshold(sblurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        sotsu = cv2.morphologyEx(sotsu, cv2.MORPH_CLOSE, skernel, iterations=2)
        sc2, _ = cv2.findContours(sotsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sall_contours.extend(sc2)

        sall_contours = sorted(sall_contours, key=cv2.contourArea, reverse=True)
        
        # Take the largest contour in side view as the object to measure height
        for contour in sall_contours:
            sarea = cv2.contourArea(contour)
            if sarea > sw * sh * 0.005:
                sx, sy, sbw, sbh = cv2.boundingRect(contour)
                # The vertical height of the bounding box represents the physical height 
                actual_height_cm = round(max(0.3, sbh * scm_per_px), 1)
                
                sannotated = side_image.copy()
                cv2.rectangle(sannotated, (sx, sy), (sx + sbw, sy + sbh), (0, 255, 0), 4)
                
                slabel = f"True Height: {actual_height_cm}cm"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (stw, sth_), _ = cv2.getTextSize(slabel, font, 0.8, 2)
                cv2.rectangle(sannotated, (sx, sy - sth_ - 15), (sx + stw + 10, sy), (0, 160, 0), -1)
                cv2.putText(sannotated, slabel, (sx + 5, sy - 7), font, 0.8, (255, 255, 255), 2)
                
                # Encode side image
                _, sbuffer = cv2.imencode(".jpg", sannotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                processed_side_image = base64.b64encode(sbuffer).decode("utf-8")
                break

    # Object detection (Top View)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)

    kernel = np.ones((5, 5), np.uint8)
    all_contours = []

    # Edge detection
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    c1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(c1)

    # Otsu threshold
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    c2, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(c2)

    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

    min_area = w * h * 0.005
    max_area = w * h * 0.99  # Allow very large objects
    detected_regions = []
    measured_objects = []
    annotated = image.copy()

    for contour in all_contours[:30]:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 15:  # more relaxed aspect ratio
            continue

        # More relaxed margin
        margin = 0
        if x < margin or y < margin or x + bw > w - margin or y + bh > h - margin:
            continue

        # Duplicate check
        is_dup = False
        for rx, ry, rw, rh in detected_regions:
            ox = max(0, min(x + bw, rx + rw) - max(x, rx))
            oy = max(0, min(y + bh, ry + rh) - max(y, ry))
            overlap = ox * oy
            union = bw * bh + rw * rh - overlap
            if overlap / max(1, union) > 0.3:
                is_dup = True
                break
        if is_dup:
            continue

        detected_regions.append((x, y, bw, bh))

        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w

        length_cm = round(max(0.3, rect_w * cm_per_px), 1)
        width_cm = round(max(0.3, rect_h * cm_per_px), 1)
        if length_cm < width_cm:
            length_cm, width_cm = width_cm, length_cm

        # 3D height assignment
        height_cm = None
        if mode == "3d":
            if actual_height_cm is not None:
                height_cm = actual_height_cm
            else:
                # Estimate if side view not provided
                roi = gray[y:y+bh, x:x+bw]
                if roi.size > 0:
                    texture_var = np.var(roi)
                    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                    grad_mean = np.mean(np.sqrt(sobelx**2 + sobely**2))
                    depth_factor = min(1.0, (texture_var / 2000) * 0.3 + (grad_mean / 50) * 0.7)
                    min_dim = min(length_cm, width_cm)
                    height_cm = round(max(0.3, min_dim * (0.2 + depth_factor * 0.5)), 1)

        obj_id = len(measured_objects) + 1
        obj_type = "3D" if height_cm else "2D"
        color = (0, 255, 0) if obj_type == "3D" else (255, 165, 0)
        bg_color = (0, 160, 0) if obj_type == "3D" else (200, 130, 0)

        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        label = f"{obj_type}: Object {obj_id}"
        if height_cm:
            dims = f"L:{length_cm} W:{width_cm} H:{height_cm} cm"
        else:
            dims = f"L:{length_cm} x W:{width_cm} cm"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th_), _ = cv2.getTextSize(label, font, 0.55, 2)
        cv2.rectangle(annotated, (x, y - th_ - 10), (x + tw + 8, y), bg_color, -1)
        cv2.putText(annotated, label, (x + 4, y - 5), font, 0.55, (255, 255, 255), 2)

        (dw, dh), _ = cv2.getTextSize(dims, font, 0.55, 2)
        cv2.rectangle(annotated, (x, y + bh), (x + dw + 8, y + bh + dh + 10), bg_color, -1)
        cv2.putText(annotated, dims, (x + 4, y + bh + dh + 5), font, 0.55, (255, 255, 255), 2)

        measured_objects.append(
            MeasuredObject(
                object_id=obj_id,
                object_type=obj_type,
                label=f"Object {obj_id}",
                confidence=0.7,
                length_cm=length_cm,
                width_cm=width_cm,
                height_cm=height_cm,
                bounding_box=(x, y, bw, bh),
                center=(x + bw // 2, y + bh // 2),
            )
        )

        if len(measured_objects) >= 10:
            break

    # Status
    cv2.putText(annotated, f"Mode: {mode.upper()} | Top Dist: {camera_distance_cm}cm", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if actual_height_cm:
        cv2.putText(annotated, f"Side Dist: {side_camera_distance_cm}cm (True 3D)", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    processed_image = base64.b64encode(buffer).decode("utf-8")

    msg = f"Measured {len(measured_objects)} object(s)" if measured_objects else "No objects detected"

    return MeasurementResult(
        success=True,
        message=msg,
        objects=measured_objects,
        reference_detected=False,
        processed_image_base64=processed_image,
        processed_side_image_base64=processed_side_image,
        mode=mode,
        calibration_info={
            "method": "local_opencv_2_views" if actual_height_cm else "local_opencv",
            "camera_distance_cm": camera_distance_cm,
            "side_camera_distance_cm": side_camera_distance_cm
        },
    )


def measure_objects(
    image: np.ndarray,
    mode: str = "2d",
    camera_distance_cm: float = 30.0,
    image_base64: Optional[str] = None,
    side_image: Optional[np.ndarray] = None,
    side_camera_distance_cm: float = 30.0,
) -> MeasurementResult:
    """
    Main measurement function. Always use local 2-view processing for perfection.
    """
    # Overriding HF space completely in favor of robust 2-view OpenCV approach
    return measure_objects_local(image, mode, camera_distance_cm, side_image, side_camera_distance_cm)
