import cv2
import numpy as np
from app.utils.image_processing import measure_objects_local
import traceback

try:
    img_path = '/home/lokesh/object-measure-app/screenshot/WhatsApp Image 2026-03-22 at 19.04.25.jpeg'
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
    else:
        print("Image loaded, shape:", img.shape)
        # We need to simulate the UI having distance = 65
        res = measure_objects_local(img, mode='3d', camera_distance_cm=65.0)
        print("Success:", res.success)
        print("Message:", res.message)
        print("Objects detected:", len(res.objects))
        for ob in res.objects:
            print(f"- {ob.label} ({ob.object_type}): L={ob.length_cm} W={ob.width_cm} H={ob.height_cm}")
            print(f"  Confidence: {ob.confidence}")
            print(f"  Bounding Box: {ob.bounding_box}")
except Exception as e:
    traceback.print_exc()
