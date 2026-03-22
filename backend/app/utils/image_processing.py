"""
Image Processing Utilities for Object Size Measurement
Uses Google Gemini Vision AI for intelligent object detection and measurement
Falls back to local OpenCV processing if Gemini is unavailable
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import base64
import json
import logging
import os
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAsVcu6NemLvN2f76Me2xAQfsMl4Ezttjs")

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


def measure_with_gemini(
    image: np.ndarray,
    mode: str = "2d",
    camera_distance_cm: float = 30.0,
    side_image: Optional[np.ndarray] = None,
    side_camera_distance_cm: float = 30.0,
) -> Optional[MeasurementResult]:
    """
    Use Google Gemini Vision AI to detect and measure objects.
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")

        h, w = image.shape[:2]

        # Encode image to base64
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Build the prompt
        if mode == "3d":
            measurement_type = "length_cm, width_cm, and height_cm"
            mode_instruction = """For 3D mode, estimate the height of each object based on visual cues like shadows, perspective, and object type.
Return height_cm for each object."""
        else:
            measurement_type = "length_cm and width_cm"
            mode_instruction = "For 2D mode, set height_cm to null."

        prompt = f"""You are an expert object measurement AI. Analyze this image and measure all distinct objects visible.

**Camera Setup:**
- Camera distance from objects: {camera_distance_cm} cm
- Image resolution: {w}x{h} pixels
- The camera is pointing at objects from above (bird's eye / top-down view)

**Your Task:**
1. Identify each distinct object in the image (ignore the background/table/surface)
2. Estimate the real-world dimensions ({measurement_type}) of each object in centimeters
3. Provide the bounding box location of each object in pixels [x, y, width, height]

{mode_instruction}

**Important Rules:**
- Camera distance of {camera_distance_cm}cm means the camera is {camera_distance_cm}cm away from the object
- Use the camera distance to calibrate your size estimates
- Be precise - round to 1 decimal place
- Common objects: a credit card is 8.5x5.4cm, a phone is ~15x7cm, an A4 sheet is 29.7x21cm
- If an object is very close (10cm distance), it will appear very large in the frame
- Label objects descriptively (e.g., "Phone", "Box", "Book", "Cup")

**Return ONLY valid JSON in this exact format:**
{{
  "objects": [
    {{
      "label": "Object Name",
      "confidence": 0.85,
      "length_cm": 15.2,
      "width_cm": 7.5,
      "height_cm": null,
      "bbox": [100, 150, 300, 200],
      "center": [250, 250]
    }}
  ]
}}

If no distinct objects are found, return: {{"objects": []}}
Return ONLY the JSON, no markdown, no explanation."""

        # Create the image part for Gemini
        image_part = {
            "mime_type": "image/jpeg",
            "data": img_b64
        }

        response = model.generate_content([prompt, image_part])
        response_text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(lines)

        logger.info(f"Gemini response: {response_text[:200]}")

        # Parse JSON
        data = json.loads(response_text)
        objects_data = data.get("objects", [])

        # Build measured objects and annotated image
        measured_objects = []
        annotated = image.copy()

        for i, obj in enumerate(objects_data):
            obj_id = i + 1
            label = obj.get("label", f"Object {obj_id}")
            confidence = obj.get("confidence", 0.8)
            length_cm = round(float(obj.get("length_cm", 0)), 1)
            width_cm = round(float(obj.get("width_cm", 0)), 1)
            height_cm = obj.get("height_cm")
            if height_cm is not None:
                height_cm = round(float(height_cm), 1)
            
            # Ensure length >= width
            if length_cm < width_cm:
                length_cm, width_cm = width_cm, length_cm

            bbox = obj.get("bbox", [0, 0, 100, 100])
            if len(bbox) == 4:
                x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            else:
                x, y, bw, bh = 0, 0, 100, 100

            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(10, min(bw, w - x))
            bh = max(10, min(bh, h - y))

            center = obj.get("center", [x + bw // 2, y + bh // 2])
            cx, cy = int(center[0]), int(center[1])

            obj_type = "3D" if height_cm else "2D"
            color = (0, 255, 0) if obj_type == "3D" else (255, 165, 0)
            bg_color = (0, 160, 0) if obj_type == "3D" else (200, 130, 0)

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

            # Draw label
            lbl = f"{obj_type}: {label}"
            if height_cm:
                dims = f"L:{length_cm} W:{width_cm} H:{height_cm} cm"
            else:
                dims = f"L:{length_cm} x W:{width_cm} cm"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th_), _ = cv2.getTextSize(lbl, font, 0.55, 2)
            cv2.rectangle(annotated, (x, y - th_ - 10), (x + tw + 8, y), bg_color, -1)
            cv2.putText(annotated, lbl, (x + 4, y - 5), font, 0.55, (255, 255, 255), 2)

            (dw, dh), _ = cv2.getTextSize(dims, font, 0.55, 2)
            cv2.rectangle(annotated, (x, y + bh), (x + dw + 8, y + bh + dh + 10), bg_color, -1)
            cv2.putText(annotated, dims, (x + 4, y + bh + dh + 5), font, 0.55, (255, 255, 255), 2)

            measured_objects.append(
                MeasuredObject(
                    object_id=obj_id,
                    object_type=obj_type,
                    label=label,
                    confidence=confidence,
                    length_cm=length_cm,
                    width_cm=width_cm,
                    height_cm=height_cm,
                    bounding_box=(x, y, bw, bh),
                    center=(cx, cy),
                )
            )

        # Status text
        cv2.putText(annotated, f"Mode: {mode.upper()} | Dist: {camera_distance_cm}cm (Gemini AI)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, out_buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_image = base64.b64encode(out_buffer).decode("utf-8")

        # Handle side image for 3D
        processed_side_image = None
        if mode == "3d" and side_image is not None:
            side_annotated = side_image.copy()
            cv2.putText(side_annotated, f"Side View | Dist: {side_camera_distance_cm}cm", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            _, side_buffer = cv2.imencode(".jpg", side_annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            processed_side_image = base64.b64encode(side_buffer).decode("utf-8")

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
                "method": "gemini_vision_ai",
                "camera_distance_cm": camera_distance_cm,
                "model": "gemini-2.0-flash",
            },
        )

    except Exception as e:
        logger.error(f"Gemini error: {e}", exc_info=True)
        return None  # Fall back to local


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
        
        for contour in sall_contours:
            sarea = cv2.contourArea(contour)
            if sarea > sw * sh * 0.005:
                sx, sy, sbw, sbh = cv2.boundingRect(contour)
                actual_height_cm = round(max(0.3, sbh * scm_per_px), 1)
                break

    # Object detection (Top View)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1)

    kernel = np.ones((5, 5), np.uint8)
    all_contours = []

    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    c1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(c1)

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    c2, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(c2)

    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

    min_area = w * h * 0.005
    max_area = w * h * 0.99
    detected_regions = []
    measured_objects = []
    annotated = image.copy()

    for contour in all_contours[:30]:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 15:
            continue

        margin = 0
        if x < margin or y < margin or x + bw > w - margin or y + bh > h - margin:
            continue

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

        height_cm = None
        if mode == "3d":
            if actual_height_cm is not None:
                height_cm = actual_height_cm
            else:
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
    cv2.putText(annotated, f"Mode: {mode.upper()} | Dist: {camera_distance_cm}cm (Local Fallback)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    processed_image = base64.b64encode(buffer).decode("utf-8")

    msg = f"Measured {len(measured_objects)} object(s)" if measured_objects else "No objects detected"

    return MeasurementResult(
        success=True,
        message=msg,
        objects=measured_objects,
        reference_detected=False,
        processed_image_base64=processed_image,
        processed_side_image_base64=None,
        mode=mode,
        calibration_info={
            "method": "local_opencv_fallback",
            "camera_distance_cm": camera_distance_cm,
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
    Main measurement function. 
    Uses Gemini AI first, falls back to local OpenCV if Gemini fails.
    """
    # Try Gemini first
    result = measure_with_gemini(image, mode, camera_distance_cm, side_image, side_camera_distance_cm)
    if result is not None:
        return result

    # Fallback to local OpenCV
    logger.warning("Gemini failed, using local OpenCV fallback")
    return measure_objects_local(image, mode, camera_distance_cm, side_image, side_camera_distance_cm)
