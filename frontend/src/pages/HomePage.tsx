import React, { useState, useCallback, useEffect, useRef } from 'react';
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
  IonFabButton,
  IonSegment,
  IonSegmentButton,
  IonLabel,
} from '@ionic/react';
import {
  camera,
  videocam,
  refreshOutline,
  checkmarkCircle,
  alertCircle,
  cubeOutline,
  squareOutline,
  resizeOutline,
  close,
  cardOutline,
  documentOutline,
  checkmark,
  warning,
} from 'ionicons/icons';
import { CameraPreview, CameraPreviewOptions } from '@capacitor-community/camera-preview';
import {
  measureRealtime,
  RealtimeMeasurementResponse,
  MeasuredObject3D,
  CalibrationInfo,
  ReferenceType,
  checkHealth,
  warmupModels,
  calibrateMeasurement,
} from '../services/api';

const HomePage: React.FC = () => {
  // State
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
  const [measurements, setMeasurements] = useState<MeasuredObject3D[]>([]);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [apiReady, setApiReady] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [liveResults, setLiveResults] = useState<MeasuredObject3D[]>([]);
  const [calibrationInfo, setCalibrationInfo] = useState<CalibrationInfo | null>(null);
  const [referenceType, setReferenceType] = useState<ReferenceType>('credit_card');
  
  const liveIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isCapturingRef = useRef(false);

  // Initialize and warm up models
  useEffect(() => {
    const init = async () => {
      // Check API health
      const healthy = await checkHealth();
      setApiReady(healthy);
      
      if (healthy) {
        // Warm up the processor
        setIsWarmingUp(true);
        try {
          await warmupModels();
          // Set default calibration
          await calibrateMeasurement(referenceType);
        } catch (e) {
          console.log('Warmup failed:', e);
        }
        setIsWarmingUp(false);
      }
    };
    
    init();
    
    // Cleanup on unmount
    return () => {
      stopLiveMode();
    };
  }, []);

  // Handle reference type change
  const handleReferenceChange = async (newType: ReferenceType) => {
    setReferenceType(newType);
    try {
      await calibrateMeasurement(newType);
    } catch (e) {
      console.log('Calibration update failed:', e);
    }
  };

  // Start live camera preview
  const startLiveMode = async () => {
    try {
      const cameraPreviewOptions: CameraPreviewOptions = {
        position: 'rear',
        parent: 'cameraPreview',
        className: 'cameraPreview',
        toBack: false,
        width: window.innerWidth,
        height: window.innerHeight - 200,
        enableZoom: true,
        enableHighResolution: true,
        disableAudio: true,
      };

      await CameraPreview.start(cameraPreviewOptions);
      setIsLiveMode(true);
      setMeasurements([]);
      setAnnotatedImage(null);
      setErrorMessage(null);
      setCapturedImage(null);
      setLiveResults([]);
      setCalibrationInfo(null);
      
      // Start continuous capture every 1.5 seconds
      liveIntervalRef.current = setInterval(async () => {
        if (!isCapturingRef.current) {
          await captureAndProcess();
        }
      }, 1500);
      
    } catch (error) {
      console.error('Failed to start camera:', error);
      setErrorMessage('Failed to start camera. Please check permissions.');
    }
  };

  // Stop live camera preview
  const stopLiveMode = async () => {
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
    
    try {
      await CameraPreview.stop();
    } catch (e) {
      console.log('Camera stop error:', e);
    }
    
    setIsLiveMode(false);
  };

  // Capture frame and process
  const captureAndProcess = async () => {
    if (isCapturingRef.current) return;
    isCapturingRef.current = true;
    
    try {
      const result = await CameraPreview.capture({
        quality: 85,
      });
      
      if (result.value) {
        const response: RealtimeMeasurementResponse = await measureRealtime(result.value, {
          returnAnnotated: false,
        });
        
        if (response.success) {
          setLiveResults(response.objects);
          setCalibrationInfo(response.calibration_info);
          setErrorMessage(null);
        } else {
          setLiveResults([]);
        }
      }
    } catch (error) {
      console.error('Capture error:', error);
    } finally {
      isCapturingRef.current = false;
    }
  };

  // Capture single photo in live mode
  const capturePhoto = async () => {
    setIsProcessing(true);
    
    try {
      const result = await CameraPreview.capture({
        quality: 90,
      });
      
      if (result.value) {
        // Stop live mode
        await stopLiveMode();
        
        // Process with annotated image
        const startTime = Date.now();
        const response: RealtimeMeasurementResponse = await measureRealtime(result.value, {
          returnAnnotated: true,
        });
        
        setProcessingTime(response.processing_time_ms || (Date.now() - startTime));
        setCalibrationInfo(response.calibration_info);
        
        if (response.success && response.objects.length > 0) {
          setMeasurements(response.objects);
          if (response.annotated_image) {
            setAnnotatedImage(`data:image/jpeg;base64,${response.annotated_image}`);
          } else {
            setCapturedImage(`data:image/jpeg;base64,${result.value}`);
          }
        } else {
          setErrorMessage(response.message || 'No objects detected');
          setCapturedImage(`data:image/jpeg;base64,${result.value}`);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to capture';
      setErrorMessage(message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset everything
  const handleReset = useCallback(() => {
    setMeasurements([]);
    setAnnotatedImage(null);
    setErrorMessage(null);
    setCapturedImage(null);
    setLiveResults([]);
    setCalibrationInfo(null);
  }, []);

  // Render calibration status badge
  const renderCalibrationBadge = (info: CalibrationInfo | null, compact: boolean = false) => {
    if (!info) return null;
    
    const isCalibrated = info.reference_detected;
    
    if (compact) {
      return (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
          background: isCalibrated ? 'rgba(16, 185, 129, 0.9)' : 'rgba(245, 158, 11, 0.9)',
          borderRadius: '16px',
          fontSize: '12px',
          color: 'white',
        }}>
          <IonIcon icon={isCalibrated ? checkmark : warning} />
          {isCalibrated ? 'Calibrated' : 'No Reference'}
        </div>
      );
    }
    
    return (
      <div style={{
        background: isCalibrated ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
        border: `1px solid ${isCalibrated ? '#10b981' : '#f59e0b'}`,
        borderRadius: '12px',
        padding: '12px',
        marginBottom: '12px',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '6px',
        }}>
          <IonIcon 
            icon={isCalibrated ? checkmarkCircle : alertCircle} 
            style={{ color: isCalibrated ? '#10b981' : '#f59e0b', fontSize: '20px' }} 
          />
          <span style={{ 
            fontWeight: '600', 
            color: isCalibrated ? '#059669' : '#d97706',
            fontSize: '14px',
          }}>
            {isCalibrated ? 'Reference Detected' : 'No Reference Found'}
          </span>
        </div>
        <p style={{ 
          margin: 0, 
          fontSize: '12px', 
          color: '#64748b',
          lineHeight: '1.4',
        }}>
          {isCalibrated 
            ? `Using ${info.reference_type.replace('_', ' ')} (${info.pixels_per_cm.toFixed(1)} px/cm)`
            : 'Place a credit card or A4 paper next to objects for accurate measurements'
          }
        </p>
      </div>
    );
  };

  // Render measurement card
  const renderMeasurementCard = (obj: MeasuredObject3D, index: number) => {
    const is3D = obj.object_type === '3D';
    
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
              {obj.label.charAt(0).toUpperCase() + obj.label.slice(1)}
            </span>
          </div>
          <IonChip style={{ '--background': 'rgba(255,255,255,0.2)', '--color': 'white' }}>
            {is3D ? '3D' : '2D'}
          </IonChip>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '70px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ fontSize: '22px', fontWeight: '700' }}>{obj.length_cm}</div>
            <div style={{ fontSize: '11px', opacity: 0.9 }}>Length (cm)</div>
          </div>
          <div style={{ flex: 1, minWidth: '70px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '10px' }}>
            <div style={{ fontSize: '22px', fontWeight: '700' }}>{obj.breadth_cm}</div>
            <div style={{ fontSize: '11px', opacity: 0.9 }}>Breadth (cm)</div>
          </div>
          {is3D && obj.height_cm && (
            <div style={{ flex: 1, minWidth: '70px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '10px' }}>
              <div style={{ fontSize: '22px', fontWeight: '700' }}>{obj.height_cm}</div>
              <div style={{ fontSize: '11px', opacity: 0.9 }}>Height (cm)</div>
            </div>
          )}
        </div>
        
        {obj.confidence < 1 && (
          <div style={{ marginTop: '8px', fontSize: '11px', opacity: 0.7, textAlign: 'right' }}>
            Confidence: {Math.round(obj.confidence * 100)}%
          </div>
        )}
      </div>
    );
  };

  // Live overlay with measurements
  const renderLiveOverlay = () => (
    <div style={{
      position: 'absolute',
      bottom: 0,
      left: 0,
      right: 0,
      background: 'linear-gradient(transparent, rgba(0,0,0,0.85))',
      padding: '20px',
      zIndex: 100,
    }}>
      {/* Calibration status */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '12px',
      }}>
        <span style={{ color: 'white', fontSize: '12px', opacity: 0.8 }}>
          Reference: {referenceType.replace('_', ' ')}
        </span>
        {renderCalibrationBadge(calibrationInfo, true)}
      </div>
      
      {/* Live measurements */}
      {liveResults.length > 0 && (
        <div style={{ marginBottom: '16px', maxHeight: '200px', overflowY: 'auto' }}>
          {liveResults.map((obj, idx) => (
            <div key={idx} style={{
              background: obj.object_type === '3D' ? 'rgba(16, 185, 129, 0.9)' : 'rgba(59, 130, 246, 0.9)',
              borderRadius: '12px',
              padding: '12px',
              marginBottom: '8px',
              color: 'white',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: '600' }}>
                  <IonIcon icon={obj.object_type === '3D' ? cubeOutline : squareOutline} style={{ marginRight: '8px' }} />
                  {obj.label}
                </span>
                <span style={{ fontSize: '12px', opacity: 0.8 }}>{obj.object_type}</span>
              </div>
              <div style={{ display: 'flex', gap: '16px', marginTop: '8px', fontSize: '14px' }}>
                <span>L: {obj.length_cm}cm</span>
                <span>B: {obj.breadth_cm}cm</span>
                {obj.object_type === '3D' && obj.height_cm && <span>H: {obj.height_cm}cm</span>}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {liveResults.length === 0 && (
        <div style={{ textAlign: 'center', marginBottom: '16px' }}>
          <p style={{ color: 'white', margin: '0 0 8px 0', opacity: 0.9 }}>
            Point camera at objects to measure
          </p>
          {!calibrationInfo?.reference_detected && (
            <p style={{ color: '#fbbf24', margin: 0, fontSize: '12px' }}>
              Add a credit card for accurate measurements
            </p>
          )}
        </div>
      )}
      
      {/* Capture button */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '16px' }}>
        <IonFabButton 
          onClick={capturePhoto}
          disabled={isProcessing}
          style={{ '--background': '#3b82f6' }}
        >
          {isProcessing ? <IonSpinner /> : <IonIcon icon={camera} />}
        </IonFabButton>
        
        <IonFabButton 
          onClick={stopLiveMode}
          color="danger"
        >
          <IonIcon icon={close} />
        </IonFabButton>
      </div>
    </div>
  );

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
        padding: '20px',
      }}>
        <div style={{
          width: '90px',
          height: '90px',
          borderRadius: '22px',
          background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '20px',
          boxShadow: '0 15px 30px rgba(59, 130, 246, 0.3)',
        }}>
          <IonIcon icon={resizeOutline} style={{ fontSize: '45px', color: 'white' }} />
        </div>
        
        <h1 style={{
          color: '#0f172a',
          fontSize: '26px',
          fontWeight: '700',
          margin: '0 0 8px 0',
        }}>
          3D Object Measure
        </h1>
        
        <p style={{
          color: '#475569',
          fontSize: '14px',
          margin: '0 0 16px 0',
          maxWidth: '280px',
          lineHeight: '1.5',
        }}>
          Accurate 2D/3D measurements with reference calibration
        </p>

        {/* API Status */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 16px',
          background: apiReady ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
          borderRadius: '20px',
          marginBottom: '16px',
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: apiReady ? '#10b981' : '#ef4444',
          }} />
          <span style={{ fontSize: '13px', color: apiReady ? '#059669' : '#dc2626' }}>
            {isWarmingUp ? 'Initializing...' : (apiReady ? 'Ready' : 'Connecting...')}
          </span>
        </div>
      </div>

      {/* Reference Selection */}
      <div style={{
        background: 'white',
        borderRadius: '20px',
        padding: '16px',
        marginBottom: '16px',
        boxShadow: '0 8px 24px rgba(0,0,0,0.06)',
      }}>
        <h3 style={{ color: '#0f172a', margin: '0 0 12px 0', fontSize: '15px', fontWeight: '600' }}>
          Reference Object
        </h3>
        <p style={{ color: '#64748b', fontSize: '12px', margin: '0 0 12px 0' }}>
          Place this next to objects for accurate measurements
        </p>
        
        <IonSegment 
          value={referenceType} 
          onIonChange={(e) => handleReferenceChange(e.detail.value as ReferenceType)}
          style={{ '--background': '#f1f5f9' }}
        >
          <IonSegmentButton value="credit_card">
            <IonIcon icon={cardOutline} />
            <IonLabel style={{ fontSize: '11px' }}>Credit Card</IonLabel>
          </IonSegmentButton>
          <IonSegmentButton value="a4_paper">
            <IonIcon icon={documentOutline} />
            <IonLabel style={{ fontSize: '11px' }}>A4 Paper</IonLabel>
          </IonSegmentButton>
        </IonSegment>
        
        <p style={{ 
          color: '#94a3b8', 
          fontSize: '11px', 
          margin: '10px 0 0 0',
          textAlign: 'center',
        }}>
          {referenceType === 'credit_card' 
            ? 'Standard card: 8.56 x 5.4 cm' 
            : 'A4 paper: 21.0 x 29.7 cm'}
        </p>
      </div>

      {/* Instructions Card */}
      <div style={{
        background: 'white',
        borderRadius: '20px',
        padding: '16px',
        marginBottom: '16px',
        boxShadow: '0 8px 24px rgba(0,0,0,0.06)',
      }}>
        <h3 style={{ color: '#0f172a', margin: '0 0 12px 0', fontSize: '15px', fontWeight: '600' }}>
          How to get accurate measurements
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {[
            { num: '1', text: 'Place reference object next to items' },
            { num: '2', text: 'Start live measurement mode' },
            { num: '3', text: 'Hold phone ~30cm away, parallel to surface' },
            { num: '4', text: 'Wait for "Calibrated" status' },
          ].map((step) => (
            <div key={step.num} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: '#3b82f6',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '12px',
                fontWeight: '600',
                flexShrink: 0,
              }}>
                {step.num}
              </div>
              <span style={{ color: '#334155', fontSize: '13px' }}>
                {step.text}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Start Live Measurement Button */}
      <div style={{ paddingBottom: '20px' }}>
        <IonButton
          expand="block"
          size="large"
          onClick={startLiveMode}
          disabled={!apiReady || isWarmingUp}
          style={{
            '--background': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            '--color': '#ffffff',
            '--border-radius': '14px',
            '--box-shadow': '0 8px 20px rgba(16, 185, 129, 0.3)',
            height: '56px',
            fontSize: '17px',
            fontWeight: '600',
          }}
        >
          <IonIcon slot="start" icon={videocam} style={{ fontSize: '22px' }} />
          {isWarmingUp ? 'Initializing...' : 'Start Measuring'}
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

      {/* Calibration Info */}
      {renderCalibrationBadge(calibrationInfo)}

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
                {measurements.length} Object(s) Measured
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
              </div>
            </div>
          </IonCardContent>
        </IonCard>
      )}

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <IonButton
          expand="block"
          onClick={() => { handleReset(); startLiveMode(); }}
          style={{ flex: 1, '--background': '#10b981', '--border-radius': '12px', height: '48px' }}
        >
          <IonIcon slot="start" icon={videocam} />
          New Scan
        </IonButton>
        
        <IonButton
          expand="block"
          onClick={handleReset}
          fill="outline"
          style={{ flex: 1, '--border-radius': '12px', height: '48px' }}
        >
          <IonIcon slot="start" icon={refreshOutline} />
          Reset
        </IonButton>
      </div>
    </div>
  );

  // Live camera view
  const renderLiveView = () => (
    <div style={{ 
      position: 'relative', 
      width: '100%', 
      height: '100%',
      background: '#000',
    }}>
      {/* Camera preview container */}
      <div 
        id="cameraPreview" 
        style={{ 
          width: '100%', 
          height: '100%',
        }} 
      />
      
      {/* Overlay with measurements */}
      {renderLiveOverlay()}
    </div>
  );

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar style={{
          '--background': isLiveMode ? '#000000' : '#ffffff',
          '--color': isLiveMode ? '#ffffff' : '#0f172a',
        }}>
          <IonTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <IonIcon icon={resizeOutline} />
              {isLiveMode ? 'Live Measurement' : '3D Measure'}
            </div>
          </IonTitle>
        </IonToolbar>
      </IonHeader>

      <IonContent fullscreen scrollY={!isLiveMode} style={{ '--background': isLiveMode ? '#000' : '#f5f7fb' }}>
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
        {isLiveMode ? (
          renderLiveView()
        ) : measurements.length > 0 || annotatedImage || errorMessage || capturedImage ? (
          renderResultsScreen()
        ) : (
          renderLandingScreen()
        )}
      </IonContent>
    </IonPage>
  );
};

export default HomePage;
