'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { usePathname } from 'next/navigation';

const Webcam = dynamic(() => import('react-webcam'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[720px] bg-gray-200 rounded-lg flex items-center justify-center">
      카메라 로딩중...
    </div>
  ),
});

const Top5PredictionsDisplay = ({ predictions }) => {
  const [topPredictions, setTopPredictions] = useState([]);

  useEffect(() => {
    if (!predictions || predictions.length === 0) return;

    setTopPredictions(prev => {
      const allPredictions = [...prev];
      
      predictions.forEach(newPred => {
        const existingIndex = allPredictions.findIndex(p => p.action === newPred.action);
        
        if (existingIndex !== -1) {
          if (newPred.confidence > allPredictions[existingIndex].confidence) {
            allPredictions[existingIndex] = newPred;
          }
        } else {
          allPredictions.push(newPred);
        }
      });

      return allPredictions
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 5);
    });
  }, [predictions]);

  if (!topPredictions || topPredictions.length === 0) return null;

  return (
    <div className="bg-white rounded-lg shadow-lg p-4 w-64">
      <h3 className="text-lg font-bold mb-4 text-gray-800">실시간 Top 5 동작</h3>
      <div className="space-y-2">
        {topPredictions.map((pred, idx) => (
          <div 
            key={pred.action} 
            className={`flex justify-between items-center p-2 rounded ${
              idx === 0 ? 'bg-blue-100' : 'bg-gray-50'
            }`}
          >
            <span className="text-sm font-medium text-gray-700">{pred.action}</span>
            <span className="text-sm font-bold text-blue-600">
              {(pred.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const WebcamComponent = () => {
  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [objects, setObjects] = useState([]);
  const [poses, setPoses] = useState([]);
  const [fps, setFps] = useState(0);
  const [top5Predictions, setTop5Predictions] = useState([]);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisImage, setAnalysisImage] = useState(null);
  const [inputText, setInputText] = useState('');

  const processingRef = useRef(false);
  const lastTimestamp = useRef(0);
  const pathname = usePathname();

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
  
    if (pathname === '/live') {
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
  
      wsRef.current.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket Connected');
        captureFrame(); // 첫 프레임 전송
      };
  
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WebSocket raw data:', data);

        if (data.type === "skeleton") {
          setObjects(data.data.detection_results.objects || []);
          setPoses(data.data.detection_results.poses || []);
          captureFrame();
        } else if (data.type === "action" && data.data.action_result) {
          console.log('Action result received:', data.data.action_result);
          
          if (data.data.action_result.top5) {
            const predictions = data.data.action_result.top5;
            console.log('Received top5 predictions:', predictions);
            
            if (Array.isArray(predictions)) {
              setTop5Predictions(predictions);
              console.log('Top5 predictions state updated');
            }
          }
        }
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
        setTimeout(connectWebSocket, 3000); // 재연결 시도
      };
    }
  };

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [pathname]);

  const captureFrame = async () => {
    if (
      webcamRef.current &&
      wsRef.current &&
      isConnected &&
      !processingRef.current &&
      !isAnalyzing &&
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
        setAnalysisImage(imageSrc);
        
        try {
          console.log('Current top5Predictions before analysis:', top5Predictions);

          const requestData = { 
            image: imageSrc,
            text: inputText.trim(),
            top5_predictions: top5Predictions
          };

          console.log('Full request data:', {
            text: requestData.text,
            top5_predictions: requestData.top5_predictions,
            hasImage: !!requestData.image
          });

          const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData),
          });

          if (!response.ok) {
            throw new Error('Network response was not ok');
          }

          const data = await response.json();
          console.log('Analysis response:', data);
          setAnalysisResult(data);
        } catch (error) {
          console.error('Error in analysis:', error);
        } finally {
          setIsAnalyzing(false);
        }
      }
    }
  };

  useEffect(() => {
    if (!isConnected) return;  // 연결되지 않았으면 실행하지 않음

    let frameCount = 0;
    let lastFpsUpdateTime = performance.now();

    const interval = setInterval(() => {
      if (!processingRef.current) {
        captureFrame();
        frameCount++;

        const now = performance.now();
        if (now - lastFpsUpdateTime >= 1000) {
          setFps(Math.round(frameCount * 1000 / (now - lastFpsUpdateTime)));
          frameCount = 0;
          lastFpsUpdateTime = now;
        }
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isConnected]);

  useEffect(() => {
    console.log('Top5Predictions state changed:', top5Predictions);
  }, [top5Predictions]);

  return (
    <div className="container mx-auto p-4">
      <div className="flex gap-8">
        {/* 왼쪽: 웹캠 영역 */}
        <div className="flex-1">
          <div className="bg-white rounded-lg shadow-lg p-6">
            {/* 웹캠 컴포넌트 */}
            <div className="relative w-full max-w-3xl mx-auto mb-4">
              <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="w-full h-auto rounded-lg"
                audio={false}
                mirrored={true}
                videoConstraints={{
                  width: 1280,
                  height: 720,
                  facingMode: 'user',
                  frameRate: 30,
                }}
              />

              {/* 실시간 객체/포즈 감지 오버레이 */}
              {(objects.length > 0 || poses.length > 0) && (
                <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                  {objects.map((obj, idx) => (
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
                  {poses.map((pose, personIdx) => (
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

              {/* 상태 표시 */}
              <div className="absolute top-4 right-4 space-y-2">
                <div className={`px-3 py-1 rounded text-sm ${isConnected ? 'bg-green-500' : 'bg-red-500'} text-white`}>
                  {isConnected ? '연결됨' : '연결 안됨'}
                </div>
                <div className="text-sm bg-gray-500 text-white px-3 py-1 rounded">
                  FPS: {fps}
                </div>
              </div>
            </div>

            {/* 입력 및 분석 버튼 영역 */}
            <div className="max-w-3xl mx-auto">
              <div className="flex gap-4">
                <input
                  type="text"
                  className="flex-1 p-2 border rounded-lg"
                  placeholder="메시지를 입력하세요..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                />
                <button
                  className={`px-4 py-2 rounded-lg font-bold ${
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
          </div>
        </div>

        {/* 오른쪽: Top5 예측 표시 */}
        <div className="w-64">
          <Top5PredictionsDisplay predictions={top5Predictions} />
        </div>
      </div>

      {/* 분석 결과 섹션 */}
      {analysisResult && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="space-y-6 max-w-3xl mx-auto">
            {/* 분석된 이미지 */}
            <div className="relative w-full">
              <img 
                src={analysisImage} 
                alt="Analyzed frame" 
                className="w-full h-auto rounded-lg"
              />
              
              {/* 스켈톤 오버레이 */}
              {analysisResult.skeleton_data && (
                <div className="absolute top-0 left-0 w-full h-full">
                  {Object.entries(analysisResult.skeleton_data.keypoints).map(([key, kp], idx) => (
                    <div
                      key={`analysis-kp-${idx}`}
                      className="absolute w-2 h-2 bg-red-500 rounded-full"
                      style={{
                        left: `${kp.x}px`,
                        top: `${kp.y}px`,
                        opacity: kp.score > 0.5 ? 1 : 0.3,
                      }}
                    />
                  ))}
                </div>
              )}

              {/* 객체 감지 박스 오버레이 */}
              {analysisResult.object_data && analysisResult.object_data.length > 0 && (
                <>
                  {analysisResult.object_data.map((obj, idx) => (
                    <div
                      key={`analysis-obj-${idx}`}
                      className="absolute border-2 border-green-500"
                      style={{
                        left: `${obj.bbox[0]}px`,
                        top: `${obj.bbox[1]}px`,
                        width: `${obj.bbox[2] - obj.bbox[0]}px`,
                        height: `${obj.bbox[3] - obj.bbox[1]}px`,
                      }}
                    >
                      <span className="absolute -top-6 left-0 bg-green-500 text-white px-2 py-1 text-xs rounded">
                        {obj.class} ({obj.confidence?.toFixed(2)})
                      </span>
                    </div>
                  ))}
                </>
              )}
            </div>

            {/* GPT 해석 결과 표시 영역 */}
            {analysisResult?.action_result?.action && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-4 text-gray-800">분석 결과</h3>
                <p className="text-sm whitespace-pre-wrap text-gray-700">
                  {analysisResult.action_result.action}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default WebcamComponent;
