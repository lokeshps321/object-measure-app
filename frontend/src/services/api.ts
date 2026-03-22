/**
 * API configuration and service functions
 */

// API Base URL - Change this to your Render deployment URL in production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface MeasuredObject {
  object_id: number;
  object_type: string; // "2D" or "3D"
  label: string;
  confidence: number;
  length_cm: number;
  width_cm: number;
  height_cm: number | null;
  bounding_box: [number, number, number, number];
  center: [number, number] | null;
}

export interface MeasurementResponse {
  success: boolean;
  message: string;
  reference_detected: boolean;
  objects: MeasuredObject[];
  processed_image: string | null;
  processed_side_image: string | null;
  mode: string;
  calibration_info: Record<string, unknown> | null;
}

export interface ApiError {
  detail: string;
}

/**
 * Send image for measurement processing
 */
export async function measureImage(
  imageBase64: string,
  mode: string = '2d',
  cameraDistanceCm: number = 30,
  sideImageBase64?: string,
  sideCameraDistanceCm?: number
): Promise<MeasurementResponse> {
  const payload = {
    image: imageBase64,
    mode: mode,
    camera_distance_cm: cameraDistanceCm,
    side_image: sideImageBase64,
    side_camera_distance_cm: sideCameraDistanceCm
  };

  const response = await fetch(`${API_BASE_URL}/api/v1/measure/base64`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error: ApiError = await response.json();
    throw new Error(error.detail || 'Failed to process image');
  }

  return response.json();
}

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
