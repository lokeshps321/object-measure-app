/**
 * API configuration and service functions
 */

// API Base URL - Change this to your Render deployment URL in production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

export interface ApiError {
  detail: string;
}

/**
 * Send image for measurement processing
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
