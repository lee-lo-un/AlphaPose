'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';

const Webcam = dynamic(() => import('react-webcam'), {
  ssr: false,
  loading: () => <div className="w-full h-[720px] bg-gray-200 rounded-lg flex items-center justify-center">카메라 로딩중...</div>
});

const WebcamComponent = () => {
  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const processingRef = useRef(false);

  useEffect(() => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    
    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket Connected');
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetectionResult(data);
      processingRef.current = false;
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      processingRef.current = false;
    };

    wsRef.current.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket Disconnected');
      processingRef.current = false;
    };

    return () => {
      wsRef.current?.close();
    };
  }, []);

  const captureFrame = () => {
    if (webcamRef.current && wsRef.current && 
        isConnected && !isAnalyzing && 
        !processingRef.current && 
        wsRef.current.readyState === WebSocket.OPEN) {
      
      processingRef.current = true;
      const imageSrc = webcamRef.current.getScreenshot();
      
      if (imageSrc) {
        const base64Data = imageSrc.split(',')[1];
        const blob = new Blob([Uint8Array.from(atob(base64Data), c => c.charCodeAt(0))], {
          type: 'image/jpeg'
        });
        wsRef.current.send(blob);
      } else {
        processingRef.current = false;
      }
    }
  };

  useEffect(() => {
    let interval;
    if (!isAnalyzing) {
      interval = setInterval(captureFrame, 50);
    }
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isConnected, isAnalyzing]);

  return (
    <div className="relative w-full max-w-4xl">
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="rounded-lg w-full"
        audio={false}
        mirrored={true}
        videoConstraints={{
          width: 1280,
          height: 720,
          facingMode: "user",
          frameRate: 20
        }}
      />
      
      {/* 결과 표시 */}
      {detectionResult && (
        <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
          {/* 객체 감지 결과 */}
          {detectionResult.detection_results.objects.map((obj, idx) => (
            <div
              key={`box-${idx}`}
              className="absolute border-2 border-green-500"
              style={{
                left: `${obj.bbox[0]}px`,
                top: `${obj.bbox[1]}px`,
                width: `${obj.bbox[2] - obj.bbox[0]}px`,
                height: `${obj.bbox[3] - obj.bbox[1]}px`,
              }}
            >
              <span className="absolute -top-6 left-0 bg-green-500 text-white px-2 py-1 text-xs rounded">
                {obj.class} ({obj.confidence.toFixed(2)})
              </span>
            </div>
          ))}

          {/* 포즈 추정 결과 */}
          {detectionResult.detection_results.poses.map((pose, personIdx) => (
            <div key={`pose-${personIdx}`}>
              {pose.keypoints.map((kp, idx) => (
                <div
                  key={`kp-${idx}`}
                  className="absolute w-2 h-2 bg-red-500 rounded-full"
                  style={{
                    left: `${kp.coordinates.x - 2}px`,
                    top: `${kp.coordinates.y - 2}px`,
                    opacity: kp.confidence > 0.5 ? 1 : 0.3,
                  }}
                />
              ))}
            </div>
          ))}

          {/* 동작 인식 결과 */}
          {detectionResult.action_recognition && (
            <div className="absolute top-16 right-4 bg-blue-500 text-white px-4 py-2 rounded">
              동작: {detectionResult.action_recognition.action}
              <br />
              신뢰도: {(detectionResult.action_recognition.confidence * 100).toFixed(1)}%
            </div>
          )}
        </div>
      )}
      
      {/* 컨트롤 */}
      <div className="absolute top-4 right-4 space-y-2">
        <div className={`px-3 py-1 rounded text-sm ${
          isConnected ? 'bg-green-500' : 'bg-red-500'
        } text-white`}>
          {isConnected ? '연결됨' : '연결 안됨'}
        </div>
        <button
          className={`w-full px-4 py-2 rounded font-bold ${
            isAnalyzing 
              ? 'bg-gray-500 text-white'
              : 'bg-blue-500 hover:bg-blue-700 text-white'
          }`}
          onClick={() => {
            setIsAnalyzing(true);
            setTimeout(() => setIsAnalyzing(false), 1000);
          }}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? '분석 중...' : '분석'}
        </button>
      </div>
    </div>
  );
};

export default WebcamComponent; 