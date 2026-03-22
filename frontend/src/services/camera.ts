/**
 * Camera service using Capacitor Camera plugin
 */

import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { Capacitor } from '@capacitor/core';

export interface CapturedImage {
  base64String: string;
  dataUrl: string;
  format: string;
}

/**
 * Request camera permissions
 */
export async function requestCameraPermission(): Promise<boolean> {
  try {
    const permission = await Camera.requestPermissions();
    return permission.camera === 'granted';
  } catch (error) {
    console.error('Error requesting camera permission:', error);
    return false;
  }
}

/**
 * Check if camera is available
 */
export function isCameraAvailable(): boolean {
  return Capacitor.isPluginAvailable('Camera');
}

/**
 * Capture photo from camera
 */
export async function capturePhoto(): Promise<CapturedImage> {
  const image = await Camera.getPhoto({
    quality: 90,
    allowEditing: false,
    resultType: CameraResultType.Base64,
    source: CameraSource.Camera,
    width: 1920,
    height: 1080,
    correctOrientation: true,
  });

  if (!image.base64String) {
    throw new Error('Failed to capture image');
  }

  const format = image.format || 'jpeg';
  const dataUrl = `data:image/${format};base64,${image.base64String}`;

  return {
    base64String: image.base64String,
    dataUrl,
    format,
  };
}

/**
 * Pick photo from gallery
 */
export async function pickFromGallery(): Promise<CapturedImage> {
  const image = await Camera.getPhoto({
    quality: 90,
    allowEditing: false,
    resultType: CameraResultType.Base64,
    source: CameraSource.Photos,
    width: 1920,
    height: 1080,
    correctOrientation: true,
  });

  if (!image.base64String) {
    throw new Error('Failed to load image');
  }

  const format = image.format || 'jpeg';
  const dataUrl = `data:image/${format};base64,${image.base64String}`;

  return {
    base64String: image.base64String,
    dataUrl,
    format,
  };
}

/**
 * Prompt user to choose between camera and gallery
 */
export async function captureOrPick(): Promise<CapturedImage> {
  const image = await Camera.getPhoto({
    quality: 90,
    allowEditing: false,
    resultType: CameraResultType.Base64,
    source: CameraSource.Prompt,
    width: 1920,
    height: 1080,
    correctOrientation: true,
    promptLabelHeader: 'Select Image Source',
    promptLabelCancel: 'Cancel',
    promptLabelPhoto: 'From Gallery',
    promptLabelPicture: 'Take Photo',
  });

  if (!image.base64String) {
    throw new Error('Failed to get image');
  }

  const format = image.format || 'jpeg';
  const dataUrl = `data:image/${format};base64,${image.base64String}`;

  return {
    base64String: image.base64String,
    dataUrl,
    format,
  };
}
