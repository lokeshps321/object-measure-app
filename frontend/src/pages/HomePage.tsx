import React, { useState, useCallback, useEffect } from 'react';
import {
  IonPage,
  IonHeader,
  IonToolbar,
  IonTitle,
  IonContent,
  IonCard,
  IonCardContent,
  IonButton,
  IonIcon,
  IonSpinner,
  IonChip,
} from '@ionic/react';
import {
  camera,
  refreshOutline,
  checkmarkCircle,
  alertCircle,
  cubeOutline,
  squareOutline,
  resizeOutline,
  images,
} from 'ionicons/icons';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import {
  measureRealtime,
  measureImage,
  MeasurementResponse,
  RealtimeMeasurementResponse,
  MeasuredObject3D,
  MeasuredObject,
  checkHealth,
  getApiStatus,
} from '../services/api';

// Check if v2 API is available
async function isV2Available(): Promise<boolean> {
  try {
    const status = await getApiStatus();
    return status.ready === true;
  } catch {
    return false;
  }
}

const HomePage: React.FC = () => {
  // State
  const [isProcessing, setIsProcessing] = useState(false);
  const [measurements, setMeasurements] = useState<(MeasuredObject3D | MeasuredObject)[]>([]);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [apiReady, setApiReady] = useState(false);
  const [useV2Api, setUseV2Api] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);

  // Initialize
  useEffect(() => {
    const init = async () => {
      // Request camera permission
      try {
        await Camera.requestPermissions();
      } catch (e) {
        console.log('Camera permission request failed:', e);
      }
      
      // Check API health
      const healthy = await checkHealth();
      setApiReady(healthy);
      
      if (healthy) {
        // Check if v2 API is available
        const v2Available = await isV2Available();
        setUseV2Api(v2Available);
        console.log('API v2 available:', v2Available);
      }
    };
    
    init();
  }, []);

  // Capture photo using Capacitor Camera
  const capturePhoto = async (): Promise<string | null> => {
    try {
      const image = await Camera.getPhoto({
        quality: 85,
        allowEditing: false,
        resultType: CameraResultType.Base64,
        source: CameraSource.Camera,
        width: 1280,
        height: 960,
        correctOrientation: true,
      });
      return image.base64String || null;
    } catch (error) {
      console.error('Camera capture error:', error);
      return null;
    }
  };

  // Pick from gallery
  const pickFromGallery = async (): Promise<string | null> => {
    try {
      const image = await Camera.getPhoto({
        quality: 90,
        allowEditing: false,
        resultType: CameraResultType.Base64,
        source: CameraSource.Photos,
        width: 1920,
        height: 1080,
        correctOrientation: true,
      });
      return image.base64String || null;
    } catch (error) {
      console.error('Gallery pick error:', error);
      return null;
    }
  };

  // Process image with API
  const processImage = useCallback(async (base64Image: string) => {
    setIsProcessing(true);
    setErrorMessage(null);
    const startTime = Date.now();
    
    try {
      if (useV2Api) {
        // Use new v2 API with 2D/3D detection
        const result: RealtimeMeasurementResponse = await measureRealtime(base64Image, {
          returnAnnotated: true,
        });

        setProcessingTime(result.processing_time_ms || (Date.now() - startTime));
        
        if (result.success && result.objects.length > 0) {
          setMeasurements(result.objects);
          if (result.annotated_image) {
            setAnnotatedImage(`data:image/jpeg;base64,${result.annotated_image}`);
          }
        } else {
          setErrorMessage(result.message || 'No objects detected');
          setCapturedImage(`data:image/jpeg;base64,${base64Image}`);
        }
      } else {
        // Fallback to v1 API (A4 reference based)
        const result: MeasurementResponse = await measureImage(base64Image);
        
        setProcessingTime(Date.now() - startTime);
        
        if (result.success && result.objects.length > 0) {
          // Convert v1 objects to display format
          const convertedObjects: MeasuredObject3D[] = result.objects.map((obj, idx) => ({
            object_id: idx + 1,
            object_type: '2D' as const,
            label: `Object ${idx + 1}`,
            confidence: 1.0,
            length_cm: obj.width_cm,
            breadth_cm: obj.height_cm,
            height_cm: null,
            bounding_box: obj.bounding_box,
            center: [obj.bounding_box[0] + obj.bounding_box[2]/2, obj.bounding_box[1] + obj.bounding_box[3]/2] as [number, number],
            depth_value: 0,
          }));
          setMeasurements(convertedObjects);
          if (result.processed_image) {
            setAnnotatedImage(`data:image/jpeg;base64,${result.processed_image}`);
          }
        } else {
          setErrorMessage(result.message || 'No A4 paper detected. Please place objects on an A4 paper.');
          setCapturedImage(`data:image/jpeg;base64,${base64Image}`);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to process image';
      setErrorMessage(message);
      setCapturedImage(`data:image/jpeg;base64,${base64Image}`);
    } finally {
      setIsProcessing(false);
    }
  }, [useV2Api]);

  // Handle single photo capture
  const handleCapturePhoto = useCallback(async () => {
    const base64 = await capturePhoto();
    if (base64) {
      await processImage(base64);
    }
  }, [processImage]);

  // Handle gallery pick
  const handlePickGallery = useCallback(async () => {
    const base64 = await pickFromGallery();
    if (base64) {
      await processImage(base64);
    }
  }, [processImage]);

  // Reset everything
  const handleReset = useCallback(() => {
    setMeasurements([]);
    setAnnotatedImage(null);
    setErrorMessage(null);
    setCapturedImage(null);
  }, []);

  // Check if object is 3D type
  const is3DObject = (obj: MeasuredObject3D | MeasuredObject): obj is MeasuredObject3D => {
    return 'object_type' in obj;
  };

  // Render measurement card
  const renderMeasurementCard = (obj: MeasuredObject3D | MeasuredObject, index: number) => {
    const is3D = is3DObject(obj) && obj.object_type === '3D';
    const label = is3DObject(obj) ? obj.label : `Object ${index + 1}`;
    const lengthCm = is3DObject(obj) ? obj.length_cm : obj.width_cm;
    const breadthCm = is3DObject(obj) ? obj.breadth_cm : obj.height_cm;
    const heightCm = is3DObject(obj) ? obj.height_cm : null;
    const confidence = is3DObject(obj) ? obj.confidence : 1;
    
    return (
      <div
        key={index}
        style={{
          background: is3D 
            ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)' 
            : 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
          borderRadius: '16px',
          padding: '16px',
          marginBottom: '12px',
          color: 'white',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <IonIcon icon={is3D ? cubeOutline : squareOutline} style={{ fontSize: '24px' }} />
            <span style={{ fontWeight: '600', fontSize: '16px' }}>
              {label.charAt(0).toUpperCase() + label.slice(1)}
            </span>
          </div>
          <IonChip style={{ '--background': 'rgba(255,255,255,0.2)', '--color': 'white' }}>
            {is3D ? '3D' : '2D'}
          </IonChip>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '70px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ fontSize: '22px', fontWeight: '700' }}>{lengthCm}</div>
            <div style={{ fontSize: '11px', opacity: 0.9 }}>Width (cm)</div>
          </div>
          <div style={{ flex: 1, minWidth: '70px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ fontSize: '22px', fontWeight: '700' }}>{breadthCm}</div>
            <div style={{ fontSize: '11px', opacity: 0.9 }}>Height (cm)</div>
          </div>
          {is3D && heightCm && (
            <div style={{ flex: 1, minWidth: '70px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '10px' }}>
              <div style={{ fontSize: '22px', fontWeight: '700' }}>{heightCm}</div>
              <div style={{ fontSize: '11px', opacity: 0.9 }}>Depth (cm)</div>
            </div>
          )}
        </div>
        
        {confidence < 1 && (
          <div style={{ marginTop: '8px', fontSize: '11px', opacity: 0.7, textAlign: 'right' }}>
            Confidence: {Math.round(confidence * 100)}%
          </div>
        )}
      </div>
    );
  };

  // Landing screen
  const renderLandingScreen = () => (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100%',
      padding: '20px',
      background: 'linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)',
    }}>
      {/* Hero Section */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        padding: '30px 20px',
      }}>
        <div style={{
          width: '100px',
          height: '100px',
          borderRadius: '25px',
          background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '24px',
          boxShadow: '0 15px 30px rgba(59, 130, 246, 0.3)',
        }}>
          <IonIcon icon={resizeOutline} style={{ fontSize: '50px', color: 'white' }} />
        </div>
        
        <h1 style={{
          color: '#0f172a',
          fontSize: '28px',
          fontWeight: '700',
          margin: '0 0 10px 0',
        }}>
          Object Measure
        </h1>
        
        <p style={{
          color: '#475569',
          fontSize: '15px',
          margin: '0 0 16px 0',
          maxWidth: '280px',
          lineHeight: '1.5',
        }}>
          {useV2Api 
            ? 'Measure any object with AI - no reference needed!' 
            : 'Place objects on A4 paper to measure them accurately'}
        </p>

        {/* API Status */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 16px',
          background: apiReady ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
          borderRadius: '20px',
          marginBottom: '20px',
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: apiReady ? '#10b981' : '#ef4444',
          }} />
          <span style={{ fontSize: '13px', color: apiReady ? '#059669' : '#dc2626' }}>
            {apiReady ? (useV2Api ? 'AI Mode Ready' : 'Ready (A4 Mode)') : 'Connecting...'}
          </span>
        </div>
      </div>

      {/* Instructions Card */}
      <div style={{
        background: 'white',
        borderRadius: '20px',
        padding: '20px',
        marginBottom: '20px',
        boxShadow: '0 8px 24px rgba(0,0,0,0.06)',
      }}>
        <h3 style={{ color: '#0f172a', margin: '0 0 14px 0', fontSize: '16px', fontWeight: '600' }}>
          How to use
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {(useV2Api ? [
            { num: '1', text: 'Point camera at any object' },
            { num: '2', text: 'Take a photo or use live mode' },
            { num: '3', text: 'Get measurements instantly' },
          ] : [
            { num: '1', text: 'Place objects on A4 paper' },
            { num: '2', text: 'Take a photo from above' },
            { num: '3', text: 'Get measurements in cm' },
          ]).map((step) => (
            <div key={step.num} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '28px',
                height: '28px',
                borderRadius: '50%',
                background: '#3b82f6',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '14px',
                fontWeight: '600',
                flexShrink: 0,
              }}>
                {step.num}
              </div>
              <span style={{ color: '#334155', fontSize: '14px' }}>
                {step.text}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', paddingBottom: '20px' }}>
        <IonButton
          expand="block"
          size="large"
          onClick={handleCapturePhoto}
          disabled={!apiReady || isProcessing}
          style={{
            '--background': '#3b82f6',
            '--color': '#ffffff',
            '--border-radius': '14px',
            '--box-shadow': '0 8px 20px rgba(59, 130, 246, 0.25)',
            height: '54px',
            fontSize: '16px',
            fontWeight: '600',
          }}
        >
          <IonIcon slot="start" icon={camera} />
          {isProcessing ? 'Processing...' : 'Take Photo'}
        </IonButton>
        
        <IonButton
          expand="block"
          size="large"
          fill="outline"
          onClick={handlePickGallery}
          disabled={!apiReady || isProcessing}
          style={{
            '--border-radius': '14px',
            '--border-color': '#cbd5e1',
            '--color': '#0f172a',
            height: '50px',
            fontSize: '15px',
          }}
        >
          <IonIcon slot="start" icon={images} />
          Choose from Gallery
        </IonButton>
      </div>
    </div>
  );

  // Results screen
  const renderResultsScreen = () => (
    <div style={{ padding: '16px', background: '#f5f7fb', minHeight: '100%' }}>
      {/* Result Image */}
      {(annotatedImage || capturedImage) && (
        <IonCard style={{
          margin: '0 0 16px 0',
          borderRadius: '16px',
          overflow: 'hidden',
        }}>
          <img 
            src={annotatedImage || capturedImage || ''} 
            alt="Result" 
            style={{ width: '100%', display: 'block' }}
          />
        </IonCard>
      )}

      {/* Measurements */}
      {measurements.length > 0 && (
        <IonCard style={{ margin: '0 0 16px 0', borderRadius: '16px' }}>
          <IonCardContent>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '16px',
            }}>
              <IonIcon icon={checkmarkCircle} style={{ color: '#10b981', fontSize: '24px' }} />
              <h2 style={{ margin: 0, color: '#0f172a', fontSize: '18px', fontWeight: '600' }}>
                {measurements.length} Object(s) Found
              </h2>
            </div>
            
            {measurements.map((obj, idx) => renderMeasurementCard(obj, idx))}
            
            <div style={{ 
              marginTop: '12px', 
              padding: '10px', 
              background: '#f1f5f9', 
              borderRadius: '10px',
              fontSize: '13px',
              color: '#64748b',
            }}>
              Processing time: {processingTime.toFixed(0)}ms
            </div>
          </IonCardContent>
        </IonCard>
      )}

      {/* Error Message */}
      {errorMessage && (
        <IonCard style={{
          margin: '0 0 16px 0',
          borderRadius: '16px',
          background: '#fef2f2',
          border: '1px solid #fecaca',
        }}>
          <IonCardContent>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
              <IonIcon icon={alertCircle} style={{ fontSize: '28px', color: '#dc2626', flexShrink: 0 }} />
              <div>
                <h3 style={{ margin: '0 0 6px 0', color: '#991b1b', fontSize: '16px', fontWeight: '600' }}>
                  Detection Issue
                </h3>
                <p style={{ margin: 0, color: '#7f1d1d', fontSize: '14px', lineHeight: '1.4' }}>
                  {errorMessage}
                </p>
                {!useV2Api && (
                  <p style={{ margin: '8px 0 0 0', color: '#991b1b', fontSize: '13px' }}>
                    Tip: Make sure the A4 paper is fully visible and well-lit.
                  </p>
                )}
              </div>
            </div>
          </IonCardContent>
        </IonCard>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <IonButton
          expand="block"
          onClick={handleReset}
          fill="outline"
          style={{ flex: 1, '--border-radius': '12px', height: '48px' }}
        >
          <IonIcon slot="start" icon={refreshOutline} />
          Reset
        </IonButton>
        
        <IonButton
          expand="block"
          onClick={handleCapturePhoto}
          disabled={isProcessing}
          style={{ flex: 2, '--background': '#3b82f6', '--border-radius': '12px', height: '48px' }}
        >
          <IonIcon slot="start" icon={camera} />
          {isProcessing ? 'Processing...' : 'New Photo'}
        </IonButton>
      </div>
    </div>
  );

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar style={{
          '--background': '#ffffff',
          '--color': '#0f172a',
        }}>
          <IonTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <IonIcon icon={resizeOutline} />
              Object Measure
            </div>
          </IonTitle>
        </IonToolbar>
      </IonHeader>

      <IonContent fullscreen style={{ '--background': '#f5f7fb' }}>
        {/* Loading Overlay */}
        {isProcessing && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.8)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}>
            <IonSpinner name="crescent" style={{ width: '50px', height: '50px', color: '#3b82f6' }} />
            <p style={{ color: 'white', marginTop: '16px', fontSize: '15px' }}>
              Analyzing image...
            </p>
          </div>
        )}

        {/* Main Content */}
        {measurements.length > 0 || annotatedImage || errorMessage || capturedImage ? (
          renderResultsScreen()
        ) : (
          renderLandingScreen()
        )}
      </IonContent>
    </IonPage>
  );
};

export default HomePage;
