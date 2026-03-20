/**
 * Real-time camera service for live video streaming
 * Uses native camera preview with frame capture capability
 */

import { Capacitor } from '@capacitor/core';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';

export interface CapturedImage {
  base64String: string;
  dataUrl: string;
  format: string;
}

export interface CameraConfig {
  quality: number;
  width: number;
  height: number;
}

// Default camera configuration
const DEFAULT_CONFIG: CameraConfig = {
  quality: 80,
  width: 1280,
  height: 720,
};

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
export async function capturePhoto(config?: Partial<CameraConfig>): Promise<CapturedImage> {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  
  const image = await Camera.getPhoto({
    quality: finalConfig.quality,
    allowEditing: false,
    resultType: CameraResultType.Base64,
    source: CameraSource.Camera,
    width: finalConfig.width,
    height: finalConfig.height,
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

// ============== Web-based Camera Stream ==============

export interface VideoStreamConfig {
  width: number;
  height: number;
  facingMode: 'user' | 'environment';
  frameRate: number;
}

const DEFAULT_STREAM_CONFIG: VideoStreamConfig = {
  width: 1280,
  height: 720,
  facingMode: 'environment', // Back camera
  frameRate: 30,
};

/**
 * Class to manage web-based camera streaming for real-time measurement
 */
export class CameraStream {
  private stream: MediaStream | null = null;
  private videoElement: HTMLVideoElement | null = null;
  private canvasElement: HTMLCanvasElement | null = null;
  private config: VideoStreamConfig;
  private isCapturing: boolean = false;
  private captureCallback: ((frame: CapturedImage) => void) | null = null;
  private captureInterval: number | null = null;

  constructor(config?: Partial<VideoStreamConfig>) {
    this.config = { ...DEFAULT_STREAM_CONFIG, ...config };
  }

  /**
   * Initialize camera stream
   */
  async initialize(videoElement: HTMLVideoElement): Promise<boolean> {
    try {
      this.videoElement = videoElement;
      
      // Create canvas for frame capture
      this.canvasElement = document.createElement('canvas');
      this.canvasElement.width = this.config.width;
      this.canvasElement.height = this.config.height;

      // Request camera access
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: this.config.width },
          height: { ideal: this.config.height },
          facingMode: this.config.facingMode,
          frameRate: { ideal: this.config.frameRate },
        },
        audio: false,
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Attach stream to video element
      this.videoElement.srcObject = this.stream;
      await this.videoElement.play();

      return true;
    } catch (error) {
      console.error('Failed to initialize camera stream:', error);
      return false;
    }
  }

  /**
   * Capture a single frame from the video stream
   */
  captureFrame(): CapturedImage | null {
    if (!this.videoElement || !this.canvasElement) {
      return null;
    }

    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) {
      return null;
    }

    // Update canvas size to match video
    this.canvasElement.width = this.videoElement.videoWidth || this.config.width;
    this.canvasElement.height = this.videoElement.videoHeight || this.config.height;

    // Draw current video frame to canvas
    ctx.drawImage(
      this.videoElement,
      0,
      0,
      this.canvasElement.width,
      this.canvasElement.height
    );

    // Convert to base64
    const dataUrl = this.canvasElement.toDataURL('image/jpeg', 0.8);
    const base64String = dataUrl.split(',')[1];

    return {
      base64String,
      dataUrl,
      format: 'jpeg',
    };
  }

  /**
   * Start continuous frame capture
   * @param callback - Function to call with each captured frame
   * @param intervalMs - Interval between captures in milliseconds
   */
  startContinuousCapture(
    callback: (frame: CapturedImage) => void,
    intervalMs: number = 500
  ): void {
    if (this.isCapturing) {
      this.stopContinuousCapture();
    }

    this.isCapturing = true;
    this.captureCallback = callback;

    this.captureInterval = window.setInterval(() => {
      if (!this.isCapturing) return;

      const frame = this.captureFrame();
      if (frame && this.captureCallback) {
        this.captureCallback(frame);
      }
    }, intervalMs);
  }

  /**
   * Stop continuous frame capture
   */
  stopContinuousCapture(): void {
    this.isCapturing = false;
    this.captureCallback = null;

    if (this.captureInterval) {
      clearInterval(this.captureInterval);
      this.captureInterval = null;
    }
  }

  /**
   * Switch between front and back camera
   */
  async switchCamera(): Promise<boolean> {
    this.config.facingMode = this.config.facingMode === 'environment' ? 'user' : 'environment';
    
    // Stop current stream
    this.stop();
    
    // Reinitialize with new facing mode
    if (this.videoElement) {
      return this.initialize(this.videoElement);
    }
    
    return false;
  }

  /**
   * Check if stream is active
   */
  isActive(): boolean {
    return this.stream !== null && this.stream.active;
  }

  /**
   * Get current video dimensions
   */
  getDimensions(): { width: number; height: number } {
    if (this.videoElement) {
      return {
        width: this.videoElement.videoWidth || this.config.width,
        height: this.videoElement.videoHeight || this.config.height,
      };
    }
    return { width: this.config.width, height: this.config.height };
  }

  /**
   * Stop and cleanup the camera stream
   */
  stop(): void {
    this.stopContinuousCapture();

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }
  }
}

/**
 * Check if web camera streaming is supported
 */
export function isWebCameraSupported(): boolean {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Get available cameras
 */
export async function getAvailableCameras(): Promise<MediaDeviceInfo[]> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(device => device.kind === 'videoinput');
  } catch {
    return [];
  }
}
