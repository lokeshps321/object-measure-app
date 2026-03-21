# 3D Object Measurement App

A production-ready mobile app that measures real-world object dimensions using computer vision. **Now with automatic reference calibration for accurate measurements!**

## V3 Features

- **Automatic Reference Detection**: Place a credit card or A4 paper next to objects - the app auto-detects and calibrates
- **Accurate Real-World Measurements**: Proper pixel-to-cm conversion using detected reference
- **2D & 3D Object Detection**: Automatically classifies objects as 2D (flat) or 3D (with depth)
- **Real-time Measurement**: Live camera mode with continuous object detection
- **Calibration Status**: Visual indicator shows when reference is detected

## How to Get Accurate Measurements

1. **Select reference type** in the app (Credit Card or A4 Paper)
2. **Place the reference** next to the objects you want to measure
3. **Point your camera** at the scene (~30cm away, parallel to surface)
4. **Wait for "Calibrated"** status to appear
5. **All objects** in the frame are now measured accurately!

### Reference Sizes
- **Credit Card**: 8.56 × 5.398 cm (ISO/IEC 7810 ID-1 standard)
- **A4 Paper**: 21.0 × 29.7 cm

### Without Reference
If no reference is detected, measurements are estimates based on an assumed camera distance of ~30cm. These estimates may vary by ±30% or more.

## Project Structure

```
object-measure-app/
├── backend/           # FastAPI backend with OpenCV
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── routes/           # API endpoints (v1 legacy, v2 realtime)
│   │   ├── models/           # Pydantic schemas
│   │   └── utils/            # Image processing & measurement
│   ├── requirements.txt
│   └── README.md
│
├── frontend/          # React + Ionic mobile app
│   ├── src/
│   │   ├── pages/            # UI pages
│   │   ├── services/         # API & camera services
│   │   └── theme/            # Styling
│   ├── android/              # Android native project
│   ├── capacitor.config.json
│   └── README.md
│
└── README.md
```

## Quick Start

### 1. Start Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at http://localhost:8000

### 2. Start Frontend (Development)

```bash
cd frontend

# Install dependencies
npm install

# Set API URL (create .env file)
echo "VITE_API_URL=http://localhost:8000" > .env

# Start development server
npm run dev
```

### 3. Build Android APK

Requires Java 11+ and Android SDK:

```bash
cd frontend

# Build web assets
npm run build

# Sync with Capacitor
npx cap sync android

# Build APK (requires Android Studio or command line build tools)
cd android
./gradlew assembleDebug

# APK will be at: android/app/build/outputs/apk/debug/app-debug.apk
```

Or open `frontend/android` in Android Studio and build from there.

## API Endpoints

### V2 API (Recommended - Real-time with Reference Calibration)

#### POST /api/v2/measure
Measure objects with automatic reference calibration.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "return_annotated": true,
  "calibration_distance_cm": 30
}
```

**Response:**
```json
{
  "success": true,
  "message": "Detected 2 object(s) (calibrated with credit_card)",
  "objects": [
    {
      "object_id": 1,
      "object_type": "3D",
      "label": "Object 1",
      "confidence": 0.85,
      "length_cm": 15.2,
      "breadth_cm": 10.5,
      "height_cm": 3.8,
      "bounding_box": [100, 150, 200, 180],
      "center": [200, 240],
      "depth_value": 0.45
    }
  ],
  "calibration_info": {
    "reference_detected": true,
    "reference_type": "credit_card",
    "pixels_per_cm": 42.5,
    "reference_width_cm": 8.56,
    "reference_height_cm": 5.398
  },
  "annotated_image": "base64_annotated_image"
}
```

#### POST /api/v2/calibrate
Configure reference type and calibration settings.

```json
{
  "reference_type": "credit_card",
  "reference_distance_cm": 30
}
```

### V1 API (Legacy - A4 Paper Required)

#### POST /api/v1/measure/base64
Original API requiring A4 paper as reference.

## Deployment

### Backend on Render

1. Create new Web Service on Render
2. Connect your GitHub repo
3. Set:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend Configuration

Update the API URL in your frontend:

1. Create `frontend/.env`:
```
VITE_API_URL=https://your-render-app.onrender.com
```

2. Rebuild and deploy the frontend

## Technical Details

### Measurement Algorithm (V3)

1. **Reference Detection**
   - Multi-scale edge detection (Canny with different thresholds)
   - 4-corner polygon detection with aspect ratio filtering
   - Aspect ratio matching for credit card (1.586) or A4 (1.414)

2. **Calibration**
   - Calculate pixels-per-cm from detected reference dimensions
   - Optional homography matrix for perspective correction

3. **Object Detection**
   - CLAHE contrast enhancement
   - Gaussian blur + Canny edge detection
   - Contour analysis with area/aspect ratio filtering

4. **3D Classification**
   - Texture variance analysis
   - Gradient magnitude (Sobel)
   - Edge density within object
   - Brightness gradient (shadow detection)

5. **Dimension Calculation**
   - Convert pixel dimensions to cm using calibration
   - Height estimation for 3D objects based on depth factor

### Requirements

- **Backend**: Python 3.11+, OpenCV, FastAPI, NumPy
- **Frontend**: Node.js 18+, React 18, Ionic 7, Capacitor 5
- **APK Build**: Java 11+, Android SDK, Gradle 8

## Tips for Best Results

1. Use a **credit card** (everyone has one!) or **white A4 paper**
2. Ensure **good, even lighting** (avoid shadows)
3. Keep the **reference fully visible** in the frame
4. Hold camera **~30cm away** and **parallel to surface**
5. Works with any shaped objects (not just rectangular)
6. Contrasting background helps detection

## License

MIT
