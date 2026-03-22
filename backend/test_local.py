import cv2
import numpy as np
from app.utils.image_processing import measure_objects_local

# create a dummy image
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)

try:
    res = measure_objects_local(img, mode="3d", camera_distance_cm=30.0)
    print("Success:", res.success)
    print("Objects:", len(res.objects))
except Exception as e:
    import traceback
    traceback.print_exc()
