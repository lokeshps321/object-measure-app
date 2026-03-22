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
} from 'ionicons/icons';
import { useState, useCallback, useEffect } from 'react';
import { capturePhoto, pickFromGallery, requestCameraPermission } from '../services/camera';
import { measureImage, MeasurementResponse, MeasuredObject } from '../services/api';

const HomePage: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('Processing...');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [measurements, setMeasurements] = useState<MeasuredObject[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const [presentToast] = useIonToast();

  // Request camera permission on mount
  useEffect(() => {
    requestCameraPermission();
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
    setLoadingText('Detecting A4 paper...');
    setErrorMessage(null);
    setCapturedImage(dataUrl);
    setResultImage(null);
    setMeasurements([]);

    try {
      setLoadingText('Measuring objects...');
      const result: MeasurementResponse = await measureImage(base64String);

      if (result.success) {
        setMeasurements(result.objects);
        if (result.processed_image) {
          setResultImage(`data:image/jpeg;base64,${result.processed_image}`);
        }
        showToast(`Found ${result.objects.length} object(s)`, 'success');
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
  }, [showToast]);

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
        padding: '40px 20px',
      }}>
        <div style={{
          width: '120px',
          height: '120px',
          borderRadius: '30px',
          background: '#ffffff',
          border: '1px solid #e5e7eb',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '30px',
          boxShadow: '0 10px 30px rgba(15, 23, 42, 0.08)',
        }}>
          <IonIcon icon={resizeOutline} style={{ fontSize: '60px', color: '#0f172a' }} />
        </div>
        
        <h1 style={{
          color: '#0f172a',
          fontSize: '32px',
          fontWeight: '700',
          margin: '0 0 12px 0',
        }}>
          Object Measure
        </h1>
        
        <p style={{
          color: '#475569',
          fontSize: '16px',
          margin: '0 0 40px 0',
          maxWidth: '280px',
          lineHeight: '1.5',
        }}>
          Measure real objects instantly using your camera and an A4 paper reference
        </p>
      </div>

      {/* Instructions Card */}
      <div style={{
        background: '#ffffff',
        borderRadius: '20px',
        padding: '24px',
        marginBottom: '24px',
        border: '1px solid #e5e7eb',
        boxShadow: '0 10px 30px rgba(15, 23, 42, 0.06)',
      }}>
        <h3 style={{ color: '#0f172a', margin: '0 0 16px 0', fontSize: '18px' }}>
          How it works
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {[
            { num: '1', text: 'Place objects on an A4 paper' },
            { num: '2', text: 'Take a photo from above' },
            { num: '3', text: 'Get measurements in cm' },
          ].map((step) => (
            <div key={step.num} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '28px',
                height: '28px',
                borderRadius: '50%',
                background: '#0f172a',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '14px',
                fontWeight: '600',
              }}>
                {step.num}
              </div>
                <span style={{ color: '#334155', fontSize: '15px' }}>
                  {step.text}
                </span>
              </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', paddingBottom: '20px' }}>
        <IonButton
          expand="block"
          size="large"
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
                Measurements
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
                <IonChip style={{
                  '--background': '#0f172a',
                  '--color': 'white',
                  marginBottom: '12px',
                }}>
                  Object {index + 1}
                </IonChip>
                
                <div style={{ display: 'flex', gap: '20px' }}>
                  <div style={{ flex: 1, textAlign: 'center' }}>
                    <div style={{
                      fontSize: '32px',
                      fontWeight: '700',
                      color: '#0f172a',
                    }}>
                      {obj.width_cm}
                    </div>
                    <div style={{ color: '#64748b', fontSize: '13px' }}>
                      Width (cm)
                    </div>
                  </div>
                  <div style={{
                    width: '1px',
                    background: 'rgba(255,255,255,0.1)',
                  }} />
                  <div style={{ flex: 1, textAlign: 'center' }}>
                    <div style={{
                      fontSize: '32px',
                      fontWeight: '700',
                      color: '#0f172a',
                    }}>
                      {obj.height_cm}
                    </div>
                    <div style={{ color: '#64748b', fontSize: '13px' }}>
                      Height (cm)
                    </div>
                  </div>
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
              Object Measure
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
            <IonSpinner name="crescent" style={{ width: '60px', height: '60px', color: '#0f172a' }} />
            <p style={{ color: 'white', marginTop: '20px', fontSize: '16px' }}>{loadingText}</p>
          </div>
        )}

        {/* Main Content */}
        {!capturedImage && !resultImage ? renderLandingScreen() : renderResults()}
      </IonContent>
    </IonPage>
  );
};

export default HomePage;
