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
  IonRange,
  IonLabel,
  IonItem,
  useIonToast,
} from '@ionic/react';
import {
  camera,
  videocam,
  videocamOff,
  refreshOutline,
  checkmarkCircle,
  alertCircle,
  cubeOutline,
  squareOutline,
  settingsOutline,
  syncOutline,
  flashOutline,
  stopCircle,
  playCircle,
} from 'ionicons/icons';
import {
  CameraStream,
  requestCameraPermission,
  isWebCameraSupported,
  CapturedImage,
  capturePhoto,
} from '../services/camera';
import {
  measureRealtime,
  RealtimeMeasurementResponse,
  MeasuredObject3D,
  warmupModels,
  checkHealth,
} from '../services/api';

const HomePage: React.FC = () => {
  // State
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [measurements, setMeasurements] = useState<MeasuredObject3D[]>([]);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [captureInterval, setCaptureInterval] = useState<number>(1000); // ms between frames
  const [showSettings, setShowSettings] = useState(false);
  const [calibrationDistance, setCalibrationDistance] = useState<number>(100); // cm
  const [apiReady, setApiReady] = useState(false);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraStreamRef = useRef<CameraStream | null>(null);
  
  const [presentToast] = useIonToast();

  // Initialize
  useEffect(() => {
    const init = async () => {
      await requestCameraPermission();
      
      // Check API health and warm up models
      const healthy = await checkHealth();
      setApiReady(healthy);
      
      if (healthy) {
        // Warm up models in background
        warmupModels().then(result => {
          if (result.success) {
            showToast('AI models ready', 'success');
          }
        });
      }
    };
    
    init();
    
    // Cleanup on unmount
    return () => {
      if (cameraStreamRef.current) {
        cameraStreamRef.current.stop();
      }
    };
  }, []);

  const showToast = useCallback((message: string, color: 'success' | 'danger' | 'warning' = 'success') => {
    presentToast({
      message,
      duration: 2000,
      color,
      position: 'bottom',
    });
  }, [presentToast]);

  // Process a single frame
  const processFrame = useCallback(async (frame: CapturedImage) => {
    if (isProcessing) return;
    
    setIsProcessing(true);
    setErrorMessage(null);
    
    try {
      const result: RealtimeMeasurementResponse = await measureRealtime(
        frame.base64String,
        {
          returnAnnotated: true,
          calibrationDistanceCm: calibrationDistance,
        }
      );

      if (result.success) {
        setMeasurements(result.objects);
        setProcessingTime(result.processing_time_ms);
        
        if (result.annotated_image) {
          setAnnotatedImage(`data:image/jpeg;base64,${result.annotated_image}`);
        }
      } else {
        setErrorMessage(result.message);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to process frame';
      setErrorMessage(message);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, calibrationDistance]);

  // Start live camera streaming
  const startLiveMode = useCallback(async () => {
    if (!isWebCameraSupported()) {
      showToast('Camera streaming not supported on this device', 'danger');
      return;
    }

    if (!videoRef.current) return;

    try {
      // Initialize camera stream
      cameraStreamRef.current = new CameraStream({
        width: 1280,
        height: 720,
        facingMode: 'environment',
        frameRate: 30,
      });

      const success = await cameraStreamRef.current.initialize(videoRef.current);
      
      if (success) {
        setIsLiveMode(true);
        setIsStreaming(true);
        setMeasurements([]);
        setAnnotatedImage(null);
        setErrorMessage(null);
        
        // Start continuous capture
        cameraStreamRef.current.startContinuousCapture(processFrame, captureInterval);
        
        showToast('Live measurement started', 'success');
      } else {
        showToast('Failed to start camera', 'danger');
      }
    } catch (error) {
      console.error('Error starting live mode:', error);
      showToast('Camera access denied', 'danger');
    }
  }, [captureInterval, processFrame, showToast]);

  // Stop live mode
  const stopLiveMode = useCallback(() => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.stop();
      cameraStreamRef.current = null;
    }
    
    setIsLiveMode(false);
    setIsStreaming(false);
    showToast('Live measurement stopped', 'warning');
  }, [showToast]);

  // Toggle streaming (pause/resume)
  const toggleStreaming = useCallback(() => {
    if (!cameraStreamRef.current) return;
    
    if (isStreaming) {
      cameraStreamRef.current.stopContinuousCapture();
      setIsStreaming(false);
    } else {
      cameraStreamRef.current.startContinuousCapture(processFrame, captureInterval);
      setIsStreaming(true);
    }
  }, [isStreaming, captureInterval, processFrame]);

  // Capture single photo
  const handleCapturePhoto = useCallback(async () => {
    try {
      setIsProcessing(true);
      const image = await capturePhoto();
      await processFrame(image);
    } catch (error: unknown) {
      const err = error as Error;
      if (!err.message?.includes('cancel')) {
        showToast(err.message || 'Failed to capture photo', 'danger');
      }
    } finally {
      setIsProcessing(false);
    }
  }, [processFrame, showToast]);

  // Reset everything
  const handleReset = useCallback(() => {
    stopLiveMode();
    setMeasurements([]);
    setAnnotatedImage(null);
    setErrorMessage(null);
  }, [stopLiveMode]);

  // Switch camera
  const switchCamera = useCallback(async () => {
    if (cameraStreamRef.current) {
      const wasStreaming = isStreaming;
      if (wasStreaming) {
        cameraStreamRef.current.stopContinuousCapture();
      }
      
      await cameraStreamRef.current.switchCamera();
      
      if (wasStreaming) {
        cameraStreamRef.current.startContinuousCapture(processFrame, captureInterval);
      }
      
      showToast('Camera switched', 'success');
    }
  }, [isStreaming, captureInterval, processFrame, showToast]);

  // Render measurement card for an object
  const renderMeasurementCard = (obj: MeasuredObject3D) => {
    const is3D = obj.object_type === '3D';
    
    return (
      <div
        key={obj.object_id}
        style={{
          background: is3D ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)' : 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
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
            {obj.object_type}
          </IonChip>
        </div>
        
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '80px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '12px' }}>
            <div style={{ fontSize: '24px', fontWeight: '700' }}>{obj.length_cm}</div>
            <div style={{ fontSize: '12px', opacity: 0.9 }}>Length (cm)</div>
          </div>
          <div style={{ flex: 1, minWidth: '80px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '12px' }}>
            <div style={{ fontSize: '24px', fontWeight: '700' }}>{obj.breadth_cm}</div>
            <div style={{ fontSize: '12px', opacity: 0.9 }}>Breadth (cm)</div>
          </div>
          {is3D && obj.height_cm && (
            <div style={{ flex: 1, minWidth: '80px', textAlign: 'center', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '12px' }}>
              <div style={{ fontSize: '24px', fontWeight: '700' }}>{obj.height_cm}</div>
              <div style={{ fontSize: '12px', opacity: 0.9 }}>Height (cm)</div>
            </div>
          )}
        </div>
        
        <div style={{ marginTop: '8px', fontSize: '12px', opacity: 0.7, textAlign: 'right' }}>
          Confidence: {Math.round(obj.confidence * 100)}%
        </div>
      </div>
    );
  };

  // Landing screen (when not in live mode)
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
        padding: '40px 20px',
      }}>
        <div style={{
          width: '120px',
          height: '120px',
          borderRadius: '30px',
          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '30px',
          boxShadow: '0 20px 40px rgba(16, 185, 129, 0.3)',
        }}>
          <IonIcon icon={cubeOutline} style={{ fontSize: '60px', color: 'white' }} />
        </div>
        
        <h1 style={{
          color: '#0f172a',
          fontSize: '32px',
          fontWeight: '700',
          margin: '0 0 12px 0',
        }}>
          3D Measure
        </h1>
        
        <p style={{
          color: '#475569',
          fontSize: '16px',
          margin: '0 0 20px 0',
          maxWidth: '300px',
          lineHeight: '1.5',
        }}>
          Real-time object measurement with AI depth estimation. No markers needed!
        </p>

        {/* API Status */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 16px',
          background: apiReady ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
          borderRadius: '20px',
          marginBottom: '30px',
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: apiReady ? '#10b981' : '#ef4444',
          }} />
          <span style={{ fontSize: '14px', color: apiReady ? '#059669' : '#dc2626' }}>
            {apiReady ? 'AI Ready' : 'Connecting...'}
          </span>
        </div>
      </div>

      {/* Features Card */}
      <div style={{
        background: 'white',
        borderRadius: '20px',
        padding: '24px',
        marginBottom: '24px',
        boxShadow: '0 10px 30px rgba(0,0,0,0.08)',
      }}>
        <h3 style={{ color: '#0f172a', margin: '0 0 16px 0', fontSize: '18px' }}>
          How it works
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {[
            { icon: videocam, text: 'Point camera at any object', color: '#10b981' },
            { icon: cubeOutline, text: '2D: Length & Breadth', color: '#f59e0b' },
            { icon: flashOutline, text: '3D: Length, Breadth & Height', color: '#10b981' },
          ].map((step, idx) => (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{
                width: '44px',
                height: '44px',
                borderRadius: '12px',
                background: `${step.color}15`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <IonIcon icon={step.icon} style={{ fontSize: '24px', color: step.color }} />
              </div>
              <span style={{ color: '#334155', fontSize: '15px', fontWeight: '500' }}>
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
          onClick={startLiveMode}
          disabled={!apiReady}
          style={{
            '--background': 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            '--color': '#ffffff',
            '--border-radius': '16px',
            '--box-shadow': '0 10px 24px rgba(16, 185, 129, 0.3)',
            height: '60px',
            fontSize: '18px',
            fontWeight: '600',
          }}
        >
          <IonIcon slot="start" icon={videocam} />
          Start Live Measurement
        </IonButton>
        
        <IonButton
          expand="block"
          size="large"
          fill="outline"
          onClick={handleCapturePhoto}
          disabled={!apiReady}
          style={{
            '--border-radius': '16px',
            '--border-color': '#cbd5e1',
            '--color': '#0f172a',
            height: '56px',
            fontSize: '17px',
          }}
        >
          <IonIcon slot="start" icon={camera} />
          Single Photo
        </IonButton>
      </div>
    </div>
  );

  // Live measurement screen
  const renderLiveScreen = () => (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%',
      background: '#000',
    }}>
      {/* Video Preview */}
      <div style={{ 
        position: 'relative', 
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}>
        {/* Live video feed */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: annotatedImage ? 'none' : 'block',
          }}
        />
        
        {/* Annotated result overlay */}
        {annotatedImage && (
          <img
            src={annotatedImage}
            alt="Measurement result"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }}
          />
        )}
        
        {/* Processing indicator */}
        {isProcessing && (
          <div style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            background: 'rgba(0,0,0,0.7)',
            borderRadius: '20px',
            padding: '8px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <IonSpinner name="crescent" style={{ width: '20px', height: '20px', color: '#10b981' }} />
            <span style={{ color: 'white', fontSize: '14px' }}>Analyzing...</span>
          </div>
        )}
        
        {/* Stats overlay */}
        <div style={{
          position: 'absolute',
          top: '16px',
          left: '16px',
          background: 'rgba(0,0,0,0.7)',
          borderRadius: '12px',
          padding: '8px 12px',
        }}>
          <div style={{ color: 'white', fontSize: '12px' }}>
            <div>Objects: {measurements.length}</div>
            <div>Time: {processingTime.toFixed(0)}ms</div>
          </div>
        </div>
        
        {/* Camera controls */}
        <div style={{
          position: 'absolute',
          bottom: '16px',
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: '16px',
        }}>
          <IonFabButton size="small" onClick={switchCamera} style={{ '--background': 'rgba(255,255,255,0.2)' }}>
            <IonIcon icon={syncOutline} />
          </IonFabButton>
          
          <IonFabButton 
            onClick={toggleStreaming} 
            style={{ 
              '--background': isStreaming ? '#ef4444' : '#10b981',
              width: '64px',
              height: '64px',
            }}
          >
            <IonIcon icon={isStreaming ? stopCircle : playCircle} style={{ fontSize: '32px' }} />
          </IonFabButton>
          
          <IonFabButton size="small" onClick={() => setShowSettings(!showSettings)} style={{ '--background': 'rgba(255,255,255,0.2)' }}>
            <IonIcon icon={settingsOutline} />
          </IonFabButton>
        </div>
      </div>
      
      {/* Settings panel */}
      {showSettings && (
        <div style={{
          background: '#1e293b',
          padding: '16px',
          borderTopLeftRadius: '20px',
          borderTopRightRadius: '20px',
        }}>
          <IonItem lines="none" style={{ '--background': 'transparent', '--color': 'white' }}>
            <IonLabel>Capture Interval: {captureInterval}ms</IonLabel>
          </IonItem>
          <IonRange
            min={200}
            max={2000}
            step={100}
            value={captureInterval}
            onIonChange={(e) => setCaptureInterval(e.detail.value as number)}
            style={{ '--bar-background': '#475569', '--knob-background': '#10b981' }}
          />
          
          <IonItem lines="none" style={{ '--background': 'transparent', '--color': 'white' }}>
            <IonLabel>Distance: {calibrationDistance}cm</IonLabel>
          </IonItem>
          <IonRange
            min={30}
            max={300}
            step={10}
            value={calibrationDistance}
            onIonChange={(e) => setCalibrationDistance(e.detail.value as number)}
            style={{ '--bar-background': '#475569', '--knob-background': '#10b981' }}
          />
        </div>
      )}
      
      {/* Measurements panel */}
      <div style={{
        background: '#0f172a',
        borderTopLeftRadius: '24px',
        borderTopRightRadius: '24px',
        padding: '20px',
        maxHeight: '40%',
        overflowY: 'auto',
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          marginBottom: '16px',
        }}>
          <h2 style={{ color: 'white', margin: 0, fontSize: '20px', fontWeight: '600' }}>
            Measurements
          </h2>
          <IonButton fill="clear" size="small" onClick={stopLiveMode} style={{ '--color': '#ef4444' }}>
            <IonIcon slot="start" icon={videocamOff} />
            Stop
          </IonButton>
        </div>
        
        {measurements.length > 0 ? (
          measurements.map(renderMeasurementCard)
        ) : (
          <div style={{ 
            textAlign: 'center', 
            padding: '40px 20px',
            color: '#64748b',
          }}>
            <IonIcon icon={cubeOutline} style={{ fontSize: '48px', marginBottom: '12px' }} />
            <p>Point camera at objects to measure</p>
          </div>
        )}
        
        {errorMessage && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            borderRadius: '12px',
            padding: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}>
            <IonIcon icon={alertCircle} style={{ color: '#ef4444', fontSize: '24px' }} />
            <span style={{ color: '#fca5a5', fontSize: '14px' }}>{errorMessage}</span>
          </div>
        )}
      </div>
    </div>
  );

  // Results screen (for single photo mode)
  const renderResultsScreen = () => (
    <div style={{ padding: '16px', background: '#f5f7fb', minHeight: '100%' }}>
      {annotatedImage && (
        <IonCard style={{
          margin: '0 0 16px 0',
          borderRadius: '20px',
          overflow: 'hidden',
        }}>
          <img 
            src={annotatedImage} 
            alt="Measurement result" 
            style={{ width: '100%', display: 'block' }}
          />
        </IonCard>
      )}

      {measurements.length > 0 && (
        <IonCard style={{ margin: '0 0 16px 0', borderRadius: '20px' }}>
          <IonCardContent>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '20px',
            }}>
              <IonIcon icon={checkmarkCircle} style={{ color: '#10b981', fontSize: '24px' }} />
              <h2 style={{ margin: 0, color: '#0f172a', fontSize: '20px' }}>
                {measurements.length} Object(s) Measured
              </h2>
            </div>
            
            {measurements.map(renderMeasurementCard)}
            
            <div style={{ 
              marginTop: '16px', 
              padding: '12px', 
              background: '#f1f5f9', 
              borderRadius: '12px',
              fontSize: '14px',
              color: '#64748b',
            }}>
              Processing time: {processingTime.toFixed(0)}ms
            </div>
          </IonCardContent>
        </IonCard>
      )}

      {errorMessage && !measurements.length && (
        <IonCard style={{
          margin: '0 0 16px 0',
          borderRadius: '20px',
          background: '#fff1f2',
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

      <div style={{ display: 'flex', gap: '12px' }}>
        <IonButton
          expand="block"
          onClick={handleReset}
          fill="outline"
          style={{ flex: 1, '--border-radius': '14px' }}
        >
          <IonIcon slot="start" icon={refreshOutline} />
          Reset
        </IonButton>
        
        <IonButton
          expand="block"
          onClick={startLiveMode}
          style={{ flex: 2, '--background': '#10b981', '--border-radius': '14px' }}
        >
          <IonIcon slot="start" icon={videocam} />
          Live Mode
        </IonButton>
      </div>
    </div>
  );

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar style={{
          '--background': isLiveMode ? '#0f172a' : '#ffffff',
          '--color': isLiveMode ? '#ffffff' : '#0f172a',
        }}>
          <IonTitle>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <IonIcon icon={cubeOutline} />
              3D Measure
              {isLiveMode && (
                <IonChip style={{ 
                  '--background': '#10b981', 
                  '--color': 'white',
                  marginLeft: '8px',
                  fontSize: '12px',
                }}>
                  LIVE
                </IonChip>
              )}
            </div>
          </IonTitle>
        </IonToolbar>
      </IonHeader>

      <IonContent fullscreen style={{ '--background': isLiveMode ? '#000' : '#f5f7fb' }}>
        {/* Loading Overlay */}
        {isProcessing && !isLiveMode && (
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
            <IonSpinner name="crescent" style={{ width: '60px', height: '60px', color: '#10b981' }} />
            <p style={{ color: 'white', marginTop: '20px', fontSize: '16px' }}>
              Analyzing with AI...
            </p>
          </div>
        )}

        {/* Main Content */}
        {isLiveMode ? (
          renderLiveScreen()
        ) : (
          measurements.length > 0 || annotatedImage || errorMessage ? (
            renderResultsScreen()
          ) : (
            renderLandingScreen()
          )
        )}
      </IonContent>
    </IonPage>
  );
};

export default HomePage;
