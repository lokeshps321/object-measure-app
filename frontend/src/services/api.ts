/**
 * API configuration and service functions
 * V3 - Reference-based calibration for accurate measurements
 */

// API Base URL - Change this to your Render deployment URL in production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============== Types ==============

export type ObjectType = '2D' | '3D';
export type ReferenceType = 'credit_card' | 'a4_paper' | 'custom' | 'none';

export interface MeasuredObject {
  width_cm: number;
  height_cm: number;
  bounding_box: [number, number, number, number];
}

export interface MeasurementResponse {
  success: boolean;
  message: string;
  reference_detected: boolean;
  objects: MeasuredObject[];
  processed_image: string | null;
}

// New 3D measurement types
export interface MeasuredObject3D {
  object_id: number;
  object_type: ObjectType;
  label: string;
  confidence: number;
  length_cm: number;
  breadth_cm: number;
  height_cm: number | null;
  bounding_box: [number, number, number, number];
  center: [number, number];
  depth_value: number;
}

export interface CalibrationInfo {
  reference_detected: boolean;
  reference_type: ReferenceType;
  pixels_per_cm: number;
  reference_width_cm: number | null;
  reference_height_cm: number | null;
}

export interface RealtimeMeasurementResponse {
  success: boolean;
  message: string;
  objects: MeasuredObject3D[];
  frame_width: number;
  frame_height: number;
  processing_time_ms: number;
  annotated_image: string | null;
  calibration_info: CalibrationInfo | null;
}

export interface RealtimeMeasurementRequest {
  image: string;
  return_annotated?: boolean;
  calibration_distance_cm?: number;
}

export interface ApiError {
  detail: string;
}

// ============== Legacy v1 API (A4 reference) ==============

/**
 * Send image for measurement processing (legacy v1 with A4 reference)
 * @param imageBase64 - Base64 encoded image (with or without data URL prefix)
 * @returns Measurement results
 */
export async function measureImage(imageBase64: string): Promise<MeasurementResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/measure/base64`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: imageBase64 }),
  });

  if (!response.ok) {
    const error: ApiError = await response.json();
    throw new Error(error.detail || 'Failed to process image');
  }

  return response.json();
}

// ============== New v2 API (Real-time with Reference Calibration) ==============

/**
 * Real-time measurement with automatic reference calibration
 * Supports both 2D and 3D objects
 * @param imageBase64 - Base64 encoded image
 * @param options - Optional configuration
 * @returns Real-time measurement results with 2D/3D dimensions and calibration info
 */
export async function measureRealtime(
  imageBase64: string,
  options?: {
    returnAnnotated?: boolean;
    calibrationDistanceCm?: number;
  }
): Promise<RealtimeMeasurementResponse> {
  const requestBody: RealtimeMeasurementRequest = {
    image: imageBase64,
    return_annotated: options?.returnAnnotated ?? true,
    calibration_distance_cm: options?.calibrationDistanceCm,
  };

  const response = await fetch(`${API_BASE_URL}/api/v2/measure`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const error: ApiError = await response.json();
    throw new Error(error.detail || 'Failed to process image');
  }

  return response.json();
}

/**
 * Configure the measurement calibration
 * @param referenceType - Type of reference object (credit_card, a4_paper, custom)
 * @param distanceCm - Distance from camera to objects in cm
 * @param customWidth - Custom reference width (for custom type)
 * @param customHeight - Custom reference height (for custom type)
 */
export async function calibrateMeasurement(
  referenceType: ReferenceType = 'credit_card',
  distanceCm: number = 30,
  customWidth?: number,
  customHeight?: number
): Promise<{ success: boolean; message: string; scale_factor: number }> {
  const response = await fetch(`${API_BASE_URL}/api/v2/calibrate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      reference_type: referenceType,
      reference_distance_cm: distanceCm,
      reference_object_width_cm: customWidth,
      reference_object_height_cm: customHeight,
    }),
  });

  if (!response.ok) {
    throw new Error('Calibration failed');
  }

  return response.json();
}

/**
 * Warm up the backend processor for faster first inference
 */
export async function warmupModels(): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v2/warmup`, {
      method: 'POST',
    });
    return response.json();
  } catch {
    return { success: false, message: 'Warmup request failed' };
  }
}

/**
 * Get API status including calibration info
 */
export async function getApiStatus(): Promise<{
  ready: boolean;
  models_loaded: boolean;
  device: string;
  method: string;
  calibration: {
    reference_type: ReferenceType;
    default_pixels_per_cm: number;
  };
  supported_references: Array<{ type: string; size: string }>;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v2/status`);
    return response.json();
  } catch {
    return {
      ready: false,
      models_loaded: false,
      device: 'unknown',
      method: 'unknown',
      calibration: {
        reference_type: 'none',
        default_pixels_per_cm: 0,
      },
      supported_references: [],
    };
  }
}

// ============== Common ==============

/**
 * Check API health status
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get the API base URL (useful for debugging)
 */
export function getApiUrl(): string {
  return API_BASE_URL;
}
