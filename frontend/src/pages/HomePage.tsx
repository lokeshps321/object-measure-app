import React from 'react';
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
  useIonToast,
} from '@ionic/react';
import {
  camera,
  images,
  refreshOutline,
  checkmarkCircle,
  alertCircle,
  resizeOutline,
  scanOutline,
  cubeOutline,
  squareOutline,
} from 'ionicons/icons';
import { useState, useCallback, useEffect } from 'react';
import { capturePhoto, pickFromGallery, requestCameraPermission } from '../services/camera';
import { measureImage, MeasurementResponse, MeasuredObject, checkHealth } from '../services/api';

const HomePage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('Processing...');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [measurements, setMeasurements] = useState<MeasuredObject[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [mode, setMode] = useState<'2d' | '3d'>('2d');
  const [cameraDistance, setCameraDistance] = useState(30);
  const [apiReady, setApiReady] = useState(false);
  
  const [presentToast] = useIonToast();

  // Check API status on mount
  useEffect(() => {
    requestCameraPermission();
    const check = async () => {
      const ready = await checkHealth();
      setApiReady(ready);
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  const showToast = useCallback((message: string, color: 'success' | 'danger' | 'warning' = 'success') => {
    presentToast({
      message,
      duration: 3000,
      color,
      position: 'bottom',
    });
  }, [presentToast]);

  const processImage = useCallback(async (base64String: string, dataUrl: string) => {
    setIsLoading(true);
    setLoadingText(mode === '3d' ? 'AI analyzing depth...' : 'Detecting objects...');
    setErrorMessage(null);
    setCapturedImage(dataUrl);
    setResultImage(null);
    setMeasurements([]);

    try {
      setLoadingText(mode === '3d' ? 'Computing 3D measurements...' : 'Measuring objects...');
      const result: MeasurementResponse = await measureImage(base64String, mode, cameraDistance);

      if (result.success) {
        setMeasurements(result.objects);
        if (result.processed_image) {
          setResultImage(`data:image/jpeg;base64,${result.processed_image}`);
        }
        showToast(`Found ${result.objects.length} object(s)`, result.objects.length > 0 ? 'success' : 'warning');
      } else {
        setErrorMessage(result.message);
        showToast(result.message, 'warning');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to process image';
      setErrorMessage(message);
      showToast(message, 'danger');
    } finally {
      setIsLoading(false);
    }
  }, [showToast, mode, cameraDistance]);

  const handleCapturePhoto = useCallback(async () => {
    try {
      const image = await capturePhoto();
      await processImage(image.base64String, image.dataUrl);
    } catch (error: unknown) {
      const err = error as Error;
      if (err.message && (err.message.includes('cancelled') || err.message.includes('cancel'))) {
        return;
      }
      console.error('Camera error:', error);
      showToast(err.message || 'Failed to capture photo', 'danger');
    }
  }, [processImage, showToast]);

  const handlePickFromGallery = useCallback(async () => {
    try {
      const image = await pickFromGallery();
      await processImage(image.base64String, image.dataUrl);
    } catch (error: unknown) {
      const err = error as Error;
      if (err.message && (err.message.includes('cancelled') || err.message.includes('cancel'))) {
        return;
      }
      console.error('Gallery error:', error);
      showToast(err.message || 'Failed to load image', 'danger');
    }
  }, [processImage, showToast]);

  const handleReset = useCallback(() => {
    setCapturedImage(null);
    setResultImage(null);
    setMeasurements([]);
    setErrorMessage(null);
  }, []);

  // Landing screen
  const renderLandingScreen = () => (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100%',
      padding: '20px',
      background: '#f5f7fb',
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
        {/* API Status */}
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 14px',
          background: apiReady ? '#dcfce7' : '#fee2e2',
          borderRadius: '20px',
          marginBottom: '20px',
          fontSize: '13px',
        }}>
          <div style={{
            width: '8px', height: '8px', borderRadius: '50%',
            background: apiReady ? '#22c55e' : '#ef4444',
          }} />
          <span style={{ color: apiReady ? '#166534' : '#991b1b' }}>
            {apiReady ? 'AI Ready' : 'Connecting...'}
          </span>
        </div>

        <div style={{
          width: '100px',
          height: '100px',
          borderRadius: '25px',
          background: '#ffffff',
          border: '1px solid #e5e7eb',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '24px',
          boxShadow: '0 10px 30px rgba(15, 23, 42, 0.08)',
        }}>
          <IonIcon icon={resizeOutline} style={{ fontSize: '50px', color: '#0f172a' }} />
        </div>
        
        <h1 style={{
          color: '#0f172a',
          fontSize: '28px',
          fontWeight: '700',
          margin: '0 0 8px 0',
        }}>
          AI Object Measure
        </h1>
        
        <p style={{
          color: '#475569',
          fontSize: '15px',
          margin: '0 0 30px 0',
          maxWidth: '280px',
          lineHeight: '1.5',
        }}>
          Measure any object instantly — just point your camera. No reference sheet needed!
        </p>
      </div>

      {/* Mode Selection */}
      <div style={{
        background: '#ffffff',
        borderRadius: '20px',
        padding: '20px',
        marginBottom: '16px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 4px 20px rgba(15, 23, 42, 0.06)',
      }}>
        <h3 style={{ color: '#0f172a', margin: '0 0 14px 0', fontSize: '16px' }}>
          Measurement Mode
        </h3>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setMode('2d')}
            style={{
              flex: 1,
              padding: '14px',
              borderRadius: '14px',
              border: mode === '2d' ? '2px solid #3b82f6' : '2px solid #e5e7eb',
              background: mode === '2d' ? '#eff6ff' : '#ffffff',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <IonIcon icon={squareOutline} style={{ fontSize: '28px', color: mode === '2d' ? '#3b82f6' : '#94a3b8' }} />
            <span style={{ fontWeight: '600', color: mode === '2d' ? '#3b82f6' : '#64748b', fontSize: '14px' }}>2D</span>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>L × W</span>
          </button>
          <button
            onClick={() => setMode('3d')}
            style={{
              flex: 1,
              padding: '14px',
              borderRadius: '14px',
              border: mode === '3d' ? '2px solid #22c55e' : '2px solid #e5e7eb',
              background: mode === '3d' ? '#f0fdf4' : '#ffffff',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <IonIcon icon={cubeOutline} style={{ fontSize: '28px', color: mode === '3d' ? '#22c55e' : '#94a3b8' }} />
            <span style={{ fontWeight: '600', color: mode === '3d' ? '#22c55e' : '#64748b', fontSize: '14px' }}>3D</span>
            <span style={{ fontSize: '11px', color: '#94a3b8' }}>L × W × H</span>
          </button>
        </div>
      </div>

      {/* Camera Distance */}
      <div style={{
        background: '#ffffff',
        borderRadius: '20px',
        padding: '20px',
        marginBottom: '16px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 4px 20px rgba(15, 23, 42, 0.06)',
      }}>
        <h3 style={{ color: '#0f172a', margin: '0 0 10px 0', fontSize: '16px' }}>
          Camera Distance: {cameraDistance} cm
        </h3>
        <input
          type="range"
          min="10"
          max="100"
          step="5"
          value={cameraDistance}
          onChange={(e) => setCameraDistance(parseInt(e.target.value))}
          style={{ width: '100%', accentColor: '#3b82f6' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
          <span>10cm (close)</span>
          <span>100cm (far)</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', paddingBottom: '20px' }}>
        <IonButton
          expand="block"
          size="large"
          disabled={!apiReady}
          onClick={handleCapturePhoto}
          style={{
            '--background': '#0f172a',
            '--color': '#ffffff',
            '--border-radius': '16px',
            '--box-shadow': '0 10px 24px rgba(15, 23, 42, 0.15)',
            height: '56px',
            fontSize: '17px',
            fontWeight: '600',
          }}
        >
          <IonIcon slot="start" icon={camera} />
          Take Photo
        </IonButton>
        
        <IonButton
          expand="block"
          size="large"
          fill="outline"
          disabled={!apiReady}
          onClick={handlePickFromGallery}
          style={{
            '--border-radius': '16px',
            '--border-color': '#cbd5e1',
            '--color': '#0f172a',
            height: '56px',
            fontSize: '17px',
            fontWeight: '600',
          }}
        >
          <IonIcon slot="start" icon={images} />
          Choose from Gallery
        </IonButton>
      </div>
    </div>
  );

  // Results screen
  const renderResults = () => (
    <div style={{ padding: '16px', background: '#f5f7fb', minHeight: '100%' }}>
      {/* Result Image */}
      {resultImage && (
        <IonCard style={{
          margin: '0 0 16px 0',
          borderRadius: '20px',
          overflow: 'hidden',
          background: '#ffffff',
          border: '1px solid #e5e7eb',
        }}>
          <img 
            src={resultImage} 
            alt="Measurement result" 
            style={{ width: '100%', display: 'block' }}
          />
        </IonCard>
      )}

      {/* Mode Badge */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', justifyContent: 'center' }}>
        <IonChip style={{
          '--background': mode === '3d' ? '#22c55e' : '#3b82f6',
          '--color': 'white',
        }}>
          <IonIcon icon={mode === '3d' ? cubeOutline : squareOutline} />
          {mode === '3d' ? '3D Mode' : '2D Mode'}
        </IonChip>
        <IonChip style={{ '--background': '#64748b', '--color': 'white' }}>
          📏 {cameraDistance}cm distance
        </IonChip>
      </div>

      {/* Measurements List */}
      {measurements.length > 0 && (
        <IonCard style={{
          margin: '0 0 16px 0',
          borderRadius: '20px',
          background: '#ffffff',
          border: '1px solid #e5e7eb',
        }}>
          <IonCardContent>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '20px',
            }}>
              <IonIcon icon={checkmarkCircle} style={{ color: '#16a34a', fontSize: '24px' }} />
              <h2 style={{ margin: 0, color: '#0f172a', fontSize: '20px' }}>
                {measurements.length} Object(s) Measured
              </h2>
            </div>
            
            {measurements.map((obj, index) => (
              <div
                key={index}
                style={{
                  background: '#f8fafc',
                  borderRadius: '16px',
                  padding: '16px',
                  marginBottom: index < measurements.length - 1 ? '12px' : 0,
                  border: '1px solid #e2e8f0',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <IonChip style={{
                    '--background': obj.object_type === '3D' ? '#22c55e' : '#3b82f6',
                    '--color': 'white',
                    margin: 0,
                  }}>
                    {obj.label}
                  </IonChip>
                  <span style={{ 
                    padding: '4px 10px',
                    borderRadius: '8px',
                    background: obj.object_type === '3D' ? '#dcfce7' : '#dbeafe',
                    color: obj.object_type === '3D' ? '#166534' : '#1e40af',
                    fontSize: '13px',
                    fontWeight: '600',
                  }}>
                    {obj.object_type}
                  </span>
                </div>
                
                <div style={{ display: 'flex', gap: '8px' }}>
                  <div style={{ 
                    flex: 1, textAlign: 'center', padding: '10px',
                    background: '#22c55e20', borderRadius: '12px',
                  }}>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: '#0f172a' }}>
                      {obj.length_cm}
                    </div>
                    <div style={{ color: '#64748b', fontSize: '12px' }}>Length (cm)</div>
                  </div>
                  <div style={{ 
                    flex: 1, textAlign: 'center', padding: '10px',
                    background: '#3b82f620', borderRadius: '12px',
                  }}>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: '#0f172a' }}>
                      {obj.width_cm}
                    </div>
                    <div style={{ color: '#64748b', fontSize: '12px' }}>Width (cm)</div>
                  </div>
                  {obj.height_cm !== null && (
                    <div style={{ 
                      flex: 1, textAlign: 'center', padding: '10px',
                      background: '#f59e0b20', borderRadius: '12px',
                    }}>
                      <div style={{ fontSize: '28px', fontWeight: '700', color: '#0f172a' }}>
                        {obj.height_cm}
                      </div>
                      <div style={{ color: '#64748b', fontSize: '12px' }}>Height (cm)</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </IonCardContent>
        </IonCard>
      )}

      {/* Error Message */}
      {errorMessage && !measurements.length && (
        <IonCard style={{
          margin: '0 0 16px 0',
          borderRadius: '20px',
          background: '#fff1f2',
          border: '1px solid #fecdd3',
        }}>
          <IonCardContent>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <IonIcon icon={alertCircle} style={{ fontSize: '32px', color: '#dc2626' }} />
              <div>
                <h3 style={{ margin: '0 0 4px 0', color: '#0f172a' }}>Detection Failed</h3>
                <p style={{ margin: 0, color: '#475569' }}>{errorMessage}</p>
              </div>
            </div>
          </IonCardContent>
        </IonCard>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
        <IonButton
          expand="block"
          onClick={handleReset}
          fill="outline"
          style={{
            flex: 1,
            '--border-radius': '14px',
            '--border-color': '#cbd5e1',
            '--color': '#0f172a',
            height: '50px',
          }}
        >
          <IonIcon slot="start" icon={refreshOutline} />
          Reset
        </IonButton>
        
        <IonButton
          expand="block"
          onClick={handleCapturePhoto}
          style={{
            flex: 2,
            '--background': '#0f172a',
            '--color': '#ffffff',
            '--border-radius': '14px',
            height: '50px',
          }}
        >
          <IonIcon slot="start" icon={camera} />
          New Photo
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
          '--border-color': '#e5e7eb',
        }}>
          <IonTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <IonIcon icon={scanOutline} />
              AI Object Measure
            </div>
          </IonTitle>
        </IonToolbar>
      </IonHeader>

      <IonContent fullscreen style={{ '--background': '#f5f7fb' }}>
        {/* Loading Overlay */}
        {isLoading && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}>
            <IonSpinner name="crescent" style={{ width: '60px', height: '60px', color: '#3b82f6' }} />
            <p style={{ color: 'white', marginTop: '20px', fontSize: '16px' }}>{loadingText}</p>
            <p style={{ color: '#94a3b8', marginTop: '4px', fontSize: '13px' }}>
              {mode === '3d' ? 'AI is analyzing depth...' : 'Processing image...'}
            </p>
          </div>
        )}

        {/* Main Content */}
        {!capturedImage && !resultImage ? renderLandingScreen() : renderResults()}
      </IonContent>
    </IonPage>
  );
};

export default HomePage;
