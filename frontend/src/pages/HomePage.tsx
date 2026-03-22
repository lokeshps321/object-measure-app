import React, { useState, useEffect, useRef } from 'react';
import {
  IonPage,
  IonHeader,
  IonToolbar,
  IonTitle,
  IonContent,
  IonButton,
  IonIcon,
  IonSpinner,
  IonCard,
  IonCardContent,
} from '@ionic/react';
import {
  camera,
  checkmarkCircle,
  alertCircle,
  refreshOutline,
  cubeOutline,
  squareOutline,
} from 'ionicons/icons';
import { CameraPreview, CameraPreviewOptions } from '@capacitor-community/camera-preview';
import {
  measureRealtime,
  RealtimeMeasurementResponse,
  MeasuredObject3D,
  checkHealth,
} from '../services/api';

type MeasureMode = '2d' | '3d';
type CaptureStep = 'idle' | 'top' | 'side' | 'done';

const HomePage: React.FC = () => {
  // State
  const [mode, setMode] = useState<MeasureMode>('2d');
  const [step, setStep] = useState<CaptureStep>('idle');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  
  // Measurement results
  const [topViewResult, setTopViewResult] = useState<RealtimeMeasurementResponse | null>(null);
  const [_sideViewResult, setSideViewResult] = useState<RealtimeMeasurementResponse | null>(null);
  void _sideViewResult; // Used for future reference
  const [finalMeasurements, setFinalMeasurements] = useState<MeasuredObject3D[]>([]);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const cameraRef = useRef<boolean>(false);

  // Check API on mount
  useEffect(() => {
    checkHealth().then(setApiReady);
    return () => { stopCamera(); };
  }, []);

  const startCamera = async () => {
    if (cameraRef.current) return;
    
    try {
      const options: CameraPreviewOptions = {
        position: 'rear',
        parent: 'cameraPreview',
        className: 'cameraPreview',
        toBack: false,
        width: window.innerWidth,
        height: window.innerHeight * 0.6,
        enableZoom: true,
        enableHighResolution: true,
        disableAudio: true,
      };
      
      await CameraPreview.start(options);
      cameraRef.current = true;
      setIsCameraActive(true);
    } catch (error) {
      console.error('Camera error:', error);
      setErrorMessage('Failed to start camera. Check permissions.');
    }
  };

  const stopCamera = async () => {
    if (!cameraRef.current) return;
    try {
      await CameraPreview.stop();
    } catch (e) {
      console.log('Stop error:', e);
    }
    cameraRef.current = false;
    setIsCameraActive(false);
  };

  const captureAndMeasure = async (viewType: 'top' | 'side') => {
    setIsProcessing(true);
    setErrorMessage(null);
    
    try {
      const result = await CameraPreview.capture({ quality: 90 });
      
      if (!result.value) {
        throw new Error('Failed to capture image');
      }
      
      const response = await measureRealtime(result.value, {
        returnAnnotated: true,
        viewType: viewType,
      });
      
      if (viewType === 'top') {
        setTopViewResult(response);
        
        if (mode === '2d') {
          // 2D mode - done after top view
          await stopCamera();
          setFinalMeasurements(response.objects);
          if (response.annotated_image) {
            setAnnotatedImage(`data:image/jpeg;base64,${response.annotated_image}`);
          }
          setStep('done');
        } else {
          // 3D mode - need side view
          setStep('side');
        }
      } else {
        // Side view captured
        setSideViewResult(response);
        await stopCamera();
        
        // Combine top and side measurements
        if (topViewResult?.objects) {
          const combined = topViewResult.objects.map((obj, i) => {
            const sideObj = response.objects[i] || response.objects[0];
            return {
              ...obj,
              height_cm: sideObj?.height_cm || null,
              object_type: sideObj?.height_cm ? '3D' as const : obj.object_type,
            };
          });
          setFinalMeasurements(combined as MeasuredObject3D[]);
        }
        
        if (response.annotated_image) {
          setAnnotatedImage(`data:image/jpeg;base64,${response.annotated_image}`);
        }
        setStep('done');
      }
      
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Measurement failed';
      setErrorMessage(msg);
    } finally {
      setIsProcessing(false);
    }
  };

  const startMeasurement = async (selectedMode: MeasureMode) => {
    setMode(selectedMode);
    setStep('top');
    setTopViewResult(null);
    setSideViewResult(null);
    setFinalMeasurements([]);
    setAnnotatedImage(null);
    setErrorMessage(null);
    await startCamera();
  };

  const reset = async () => {
    await stopCamera();
    setStep('idle');
    setMode('2d');
    setTopViewResult(null);
    setSideViewResult(null);
    setFinalMeasurements([]);
    setAnnotatedImage(null);
    setErrorMessage(null);
  };

  // Render idle/home screen
  const renderHome = () => (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <div style={{
        width: '80px',
        height: '80px',
        margin: '40px auto 20px',
        borderRadius: '20px',
        background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <IonIcon icon={cubeOutline} style={{ fontSize: '40px', color: 'white' }} />
      </div>
      
      <h1 style={{ fontSize: '24px', margin: '0 0 10px', color: '#1e293b' }}>
        Object Measure
      </h1>
      <p style={{ color: '#64748b', marginBottom: '30px' }}>
        Accurate measurements using A4 paper
      </p>
      
      {/* API Status */}
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px 16px',
        background: apiReady ? '#dcfce7' : '#fee2e2',
        borderRadius: '20px',
        marginBottom: '30px',
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          background: apiReady ? '#22c55e' : '#ef4444',
        }} />
        <span style={{ color: apiReady ? '#166534' : '#991b1b', fontSize: '14px' }}>
          {apiReady ? 'Ready' : 'Connecting...'}
        </span>
      </div>
      
      {/* Instructions */}
      <div style={{
        background: '#f8fafc',
        borderRadius: '16px',
        padding: '20px',
        marginBottom: '24px',
        textAlign: 'left',
      }}>
        <h3 style={{ margin: '0 0 12px', fontSize: '16px', color: '#334155' }}>
          How to measure:
        </h3>
        <ol style={{ margin: 0, paddingLeft: '20px', color: '#64748b', lineHeight: '1.8' }}>
          <li>Place A4 paper on flat surface</li>
          <li>Put object ON the A4 paper</li>
          <li>Choose 2D or 3D measurement</li>
          <li>Follow camera instructions</li>
        </ol>
      </div>
      
      {/* Mode Selection Buttons */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
        <IonButton
          expand="block"
          style={{ flex: 1 }}
          disabled={!apiReady}
          onClick={() => startMeasurement('2d')}
        >
          <IonIcon slot="start" icon={squareOutline} />
          2D (L × W)
        </IonButton>
        
        <IonButton
          expand="block"
          style={{ flex: 1 }}
          color="success"
          disabled={!apiReady}
          onClick={() => startMeasurement('3d')}
        >
          <IonIcon slot="start" icon={cubeOutline} />
          3D (L × W × H)
        </IonButton>
      </div>
      
      <p style={{ fontSize: '12px', color: '#94a3b8' }}>
        3D mode requires two photos: top view + side view
      </p>
    </div>
  );

  // Render camera view for capturing
  const renderCapture = () => (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#000' }}>
      {/* Camera Preview */}
      <div 
        id="cameraPreview" 
        style={{ 
          flex: 1, 
          minHeight: '60%',
          background: '#111',
        }} 
      />
      
      {/* Instructions Panel */}
      <div style={{
        background: 'linear-gradient(transparent, rgba(0,0,0,0.9))',
        padding: '20px',
        color: 'white',
      }}>
        {/* Step Indicator */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '10px',
          marginBottom: '16px',
        }}>
          <div style={{
            padding: '6px 16px',
            borderRadius: '20px',
            background: step === 'top' ? '#3b82f6' : '#374151',
            fontSize: '14px',
          }}>
            1. Top View
          </div>
          {mode === '3d' && (
            <div style={{
              padding: '6px 16px',
              borderRadius: '20px',
              background: step === 'side' ? '#22c55e' : '#374151',
              fontSize: '14px',
            }}>
              2. Side View
            </div>
          )}
        </div>
        
        {/* Instructions */}
        <div style={{ textAlign: 'center', marginBottom: '16px' }}>
          {step === 'top' && (
            <>
              <h3 style={{ margin: '0 0 8px', fontSize: '18px' }}>
                Top View - Look Down
              </h3>
              <p style={{ margin: 0, opacity: 0.8, fontSize: '14px' }}>
                Hold phone ~30cm above, pointing straight down at the A4 paper
              </p>
            </>
          )}
          {step === 'side' && (
            <>
              <h3 style={{ margin: '0 0 8px', fontSize: '18px' }}>
                Side View - Look Horizontal
              </h3>
              <p style={{ margin: 0, opacity: 0.8, fontSize: '14px' }}>
                Now capture from the side to measure height
              </p>
            </>
          )}
        </div>
        
        {/* Capture Button */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '16px' }}>
          <IonButton
            size="large"
            shape="round"
            disabled={isProcessing || !isCameraActive}
            onClick={() => captureAndMeasure(step === 'top' ? 'top' : 'side')}
            style={{
              '--padding-start': '32px',
              '--padding-end': '32px',
            }}
          >
            {isProcessing ? (
              <IonSpinner name="crescent" />
            ) : (
              <>
                <IonIcon slot="start" icon={camera} />
                Capture {step === 'top' ? 'Top' : 'Side'}
              </>
            )}
          </IonButton>
          
          <IonButton
            size="large"
            shape="round"
            color="medium"
            onClick={reset}
          >
            Cancel
          </IonButton>
        </div>
        
        {/* Error Message */}
        {errorMessage && (
          <div style={{
            marginTop: '12px',
            padding: '10px',
            background: 'rgba(239, 68, 68, 0.2)',
            borderRadius: '8px',
            textAlign: 'center',
            color: '#fca5a5',
          }}>
            {errorMessage}
          </div>
        )}
      </div>
    </div>
  );

  // Render results
  const renderResults = () => (
    <div style={{ padding: '16px' }}>
      {/* Result Image */}
      {annotatedImage && (
        <IonCard style={{ margin: '0 0 16px', borderRadius: '16px', overflow: 'hidden' }}>
          <img src={annotatedImage} alt="Measured" style={{ width: '100%', display: 'block' }} />
        </IonCard>
      )}
      
      {/* Calibration Status */}
      {topViewResult?.calibration_info && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px',
          background: topViewResult.calibration_info.reference_detected ? '#dcfce7' : '#fef3c7',
          borderRadius: '12px',
          marginBottom: '16px',
        }}>
          <IonIcon 
            icon={topViewResult.calibration_info.reference_detected ? checkmarkCircle : alertCircle}
            style={{ 
              fontSize: '20px', 
              color: topViewResult.calibration_info.reference_detected ? '#22c55e' : '#f59e0b',
            }}
          />
          <span style={{ 
            color: topViewResult.calibration_info.reference_detected ? '#166534' : '#92400e',
            fontSize: '14px',
          }}>
            {topViewResult.calibration_info.reference_detected 
              ? `A4 Detected (${topViewResult.calibration_info.pixels_per_cm.toFixed(1)} px/cm)`
              : 'A4 paper not detected - measurements may be inaccurate'}
          </span>
        </div>
      )}
      
      {/* Measurements */}
      {finalMeasurements.length > 0 ? (
        <IonCard style={{ margin: '0 0 16px', borderRadius: '16px' }}>
          <IonCardContent>
            <h2 style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px',
              margin: '0 0 16px',
              color: '#1e293b',
            }}>
              <IonIcon icon={checkmarkCircle} style={{ color: '#22c55e' }} />
              {finalMeasurements.length} Object(s) Measured
            </h2>
            
            {finalMeasurements.map((obj, idx) => (
              <div
                key={idx}
                style={{
                  background: obj.object_type === '3D' 
                    ? 'linear-gradient(135deg, #22c55e, #16a34a)'
                    : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                  borderRadius: '12px',
                  padding: '16px',
                  marginBottom: '12px',
                  color: 'white',
                }}
              >
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  marginBottom: '12px',
                }}>
                  <span style={{ fontWeight: '600', fontSize: '16px' }}>
                    <IonIcon 
                      icon={obj.object_type === '3D' ? cubeOutline : squareOutline} 
                      style={{ marginRight: '8px' }}
                    />
                    {obj.label}
                  </span>
                  <span style={{ 
                    background: 'rgba(255,255,255,0.2)', 
                    padding: '4px 12px',
                    borderRadius: '12px',
                    fontSize: '12px',
                  }}>
                    {obj.object_type}
                  </span>
                </div>
                
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <div style={{ 
                    flex: 1, 
                    minWidth: '80px',
                    background: 'rgba(255,255,255,0.15)',
                    borderRadius: '10px',
                    padding: '12px',
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: '24px', fontWeight: '700' }}>{obj.length_cm}</div>
                    <div style={{ fontSize: '12px', opacity: 0.9 }}>Length (cm)</div>
                  </div>
                  <div style={{ 
                    flex: 1, 
                    minWidth: '80px',
                    background: 'rgba(255,255,255,0.15)',
                    borderRadius: '10px',
                    padding: '12px',
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: '24px', fontWeight: '700' }}>{obj.width_cm}</div>
                    <div style={{ fontSize: '12px', opacity: 0.9 }}>Width (cm)</div>
                  </div>
                  {obj.height_cm !== null && (
                    <div style={{ 
                      flex: 1, 
                      minWidth: '80px',
                      background: 'rgba(255,255,255,0.15)',
                      borderRadius: '10px',
                      padding: '12px',
                      textAlign: 'center',
                    }}>
                      <div style={{ fontSize: '24px', fontWeight: '700' }}>{obj.height_cm}</div>
                      <div style={{ fontSize: '12px', opacity: 0.9 }}>Height (cm)</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </IonCardContent>
        </IonCard>
      ) : (
        <IonCard style={{ margin: '0 0 16px', borderRadius: '16px' }}>
          <IonCardContent>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '12px',
              color: '#ef4444',
            }}>
              <IonIcon icon={alertCircle} style={{ fontSize: '28px' }} />
              <div>
                <h3 style={{ margin: '0 0 4px' }}>No Objects Detected</h3>
                <p style={{ margin: 0, fontSize: '14px', color: '#64748b' }}>
                  Make sure the object is placed on the A4 paper and clearly visible
                </p>
              </div>
            </div>
          </IonCardContent>
        </IonCard>
      )}
      
      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '12px' }}>
        <IonButton expand="block" style={{ flex: 1 }} onClick={() => startMeasurement(mode)}>
          <IonIcon slot="start" icon={camera} />
          Measure Again
        </IonButton>
        <IonButton expand="block" style={{ flex: 1 }} fill="outline" onClick={reset}>
          <IonIcon slot="start" icon={refreshOutline} />
          New
        </IonButton>
      </div>
    </div>
  );

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar style={{ 
          '--background': step === 'idle' || step === 'done' ? '#ffffff' : '#000000',
          '--color': step === 'idle' || step === 'done' ? '#1e293b' : '#ffffff',
        }}>
          <IonTitle>
            {step === 'idle' && 'Object Measure'}
            {step === 'top' && '1. Top View'}
            {step === 'side' && '2. Side View'}
            {step === 'done' && 'Results'}
          </IonTitle>
        </IonToolbar>
      </IonHeader>

      <IonContent fullscreen style={{ 
        '--background': step === 'idle' || step === 'done' ? '#f8fafc' : '#000000',
      }}>
        {step === 'idle' && renderHome()}
        {(step === 'top' || step === 'side') && renderCapture()}
        {step === 'done' && renderResults()}
      </IonContent>
    </IonPage>
  );
};

export default HomePage;
