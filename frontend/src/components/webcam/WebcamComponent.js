'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';

const Webcam = dynamic(() => import('react-webcam'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[720px] bg-gray-200 rounded-lg flex items-center justify-center">
      카메라 로딩중...
    </div>
  ),
});

const WebcamComponent = () => {
  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [fps, setFps] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const processingRef = useRef(false);
  const lastTimestamp = useRef(0);

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    
    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket Connected');
    };

    wsRef.current.onmessage = (event) => {
      console.log('WebSocket 메시지 수신:', event.data);
      const data = JSON.parse(event.data);
      console.log('Detection 결과 업데이트:', data); // 상태 업데이트 확인
      setDetectionResult(data);
      processingRef.current = false;
    };

    wsRef.current.onerror = () => {
      console.log('WebSocket connection error');
      setIsConnected(false);
      processingRef.current = false;
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket Disconnected');
      setIsConnected(false);
      processingRef.current = false;
      setTimeout(connectWebSocket, 3000);
    };
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const captureFrame = async () => {
    if (
      webcamRef.current &&
      wsRef.current &&
      isConnected &&
      !processingRef.current &&
      wsRef.current.readyState === WebSocket.OPEN
    ) {
      processingRef.current = true;
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const base64Data = imageSrc.split(',')[1];
        const blob = new Blob([Uint8Array.from(atob(base64Data), c => c.charCodeAt(0))], {
          type: 'image/jpeg',
        });
        wsRef.current.send(blob);
        console.log('Frame sent:', new Date().toISOString());
      } else {
        processingRef.current = false;
      }
    }
  };

  const analyzeImage = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setIsAnalyzing(true);
        try {
          const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageSrc }),
          });

          if (!response.ok) {
            throw new Error('Network response was not ok');
          }

          const data = await response.json();
          console.log('분석 결과:', data);
        } catch (error) {
          console.error('Error analyzing image:', error);
        } finally {
          setIsAnalyzing(false);
        }
      }
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      if (!processingRef.current) {
        captureFrame();
      }
    }, 30); // 30ms 간격으로 시도
    return () => clearInterval(interval);
  }, [isConnected]);

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
          facingMode: 'user',
          frameRate: 30,
        }}
      />

      {detectionResult && (
        <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
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
        </div>
      )}

      <div className="absolute top-4 right-4 space-y-2">
        <div className={`px-3 py-1 rounded text-sm ${isConnected ? 'bg-green-500' : 'bg-red-500'} text-white`}>
          {isConnected ? '연결됨' : '연결 안됨'}
        </div>
        <div className="text-sm bg-gray-500 text-white px-3 py-1 rounded">
          FPS: {fps}
        </div>
        <button
          className={`w-full px-4 py-2 rounded font-bold ${
            isAnalyzing 
              ? 'bg-gray-500 text-white'
              : 'bg-blue-500 hover:bg-blue-700 text-white'
          }`}
          onClick={analyzeImage}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? '분석 중...' : '분석'}
        </button>
      </div>
    </div>
  );
};

export default WebcamComponent;