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

const WebcamComponent = () => {
  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [objects, setObjects] = useState([]);
  const [poses, setPoses] = useState([]);
  const [fps, setFps] = useState(0);
  const [top5Predictions, setTop5Predictions] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisImage, setAnalysisImage] = useState(null);

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
        if (data.type === "skeleton") {
          // 결과를 상태로 업데이트
          setObjects(data.data.detection_results.objects || []);
          setPoses(data.data.detection_results.poses || []);
  
          // 결과를 받은 후 다음 프레임을 캡처
          captureFrame();
        } else if (data.type === "action") {
          // action_result 데이터 저장
          setTop5Predictions(data.data.top5);
        }
        processingRef.current = false; // 현재 프레임 처리 완료
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
        setAnalysisImage(imageSrc); // 캡처된 이미지 저장
        try {
          const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
              image: imageSrc,
              top5_predictions: top5Predictions
            }),
          });

          if (!response.ok) {
            throw new Error('Network response was not ok');
          }

          const data = await response.json();
          setAnalysisResult(data);
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
    }, 100); // ms 간격으로 시도

    return () => clearInterval(interval);
  }, [isConnected]);

  // 분석 결과를 표시하는 컴포넌트
  const AnalysisResultView = () => {
    if (!analysisResult) return null;

    console.log('Analysis Result:', analysisResult); // 데이터 구조 확인용

    return (
      <div className="mt-8 p-6 bg-white rounded-lg shadow-lg">
        <div className="grid grid-cols-2 gap-6">
          {/* 이미지 및 스켈레톤 표시 영역 */}
          <div className="relative">
            <img 
              src={analysisImage} 
              alt="Analyzed frame" 
              className="w-full rounded-lg"
            />
            
            {/* 스켈레톤 오버레이 */}
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

          {/* 텍스트 분석 결과 영역 */}
          <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-bold mb-2">행동 인식 결과</h3>
              {analysisResult.action_result && (
                <div>
                  <p className="text-lg font-semibold text-blue-600">
                    {analysisResult.action_result.action}
                  </p>
                  <p className="text-sm text-gray-600">
                    신뢰도: {(analysisResult.action_result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </div>

            {analysisResult.similar_actions && analysisResult.similar_actions.length > 0 && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-bold mb-2">유사한 행동들</h3>
                <ul className="space-y-2">
                  {analysisResult.similar_actions.map((action, idx) => (
                    <li key={idx} className="text-sm">
                      {action.action} (유사도: {(action.similarity * 100).toFixed(1)}%)
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {analysisResult.action_predictions && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-bold mb-2">상위 5개 예측 행동</h3>
                <ul className="space-y-2">
                  {analysisResult.action_predictions.map((pred, idx) => (
                    <li key={idx} className="text-sm">
                      {pred.action} ({(pred.confidence * 100).toFixed(1)}%)
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 행동 설명 추가 */}
            {analysisResult.action_explanation && (
                <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-lg font-bold mb-2">상세 분석</h3>
                    <div className="space-y-2">
                        <p className="text-sm">{analysisResult.action_explanation.detailed_explanation}</p>
                        {analysisResult.action_explanation.posture_description && (
                            <div>
                                <h4 className="font-semibold">자세 특징:</h4>
                                <p className="text-sm">{analysisResult.action_explanation.posture_description.main_characteristics}</p>
                            </div>
                        )}
                        {analysisResult.action_explanation.object_interactions && (
                            <div>
                                <h4 className="font-semibold">객체 상호작용:</h4>
                                <p className="text-sm">{analysisResult.action_explanation.object_interactions}</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* 예측 분석 결과 추가 */}
            {analysisResult.prediction_analysis && (
                <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-lg font-bold mb-2">예측 정확도 분석</h3>
                    <div className="space-y-2">
                        {analysisResult.prediction_analysis.match_found ? (
                            <p className="text-sm">
                                실시간 예측이 {analysisResult.prediction_analysis.ranking}번째로 
                                정확한 결과를 예측했습니다. 
                                (정확도: {(analysisResult.prediction_analysis.prediction_accuracy * 100).toFixed(1)}%)
                            </p>
                        ) : (
                            <p className="text-sm">실시간 예측과 최종 분석 결과가 일치하지 않았습니다.</p>
                        )}
                    </div>
                </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4 space-y-8">
      {/* 웹캠 영역 */}
      <div className="relative w-full">
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

        {/* 컨트롤 버튼 */}
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

      {/* 입력 및 분석 영역 */}
      <div className="w-full space-y-4">
        <div className="flex gap-4">
          <input
            type="text"
            className="flex-1 p-2 border rounded-lg"
            placeholder="메시지를 입력하세요..."
          />
          <button
            className={`px-4 py-2 rounded font-bold ${
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

        {/* GPT 해석 결과 표시 영역 */}
        {analysisResult?.gpt_interpretation && (
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-bold mb-2">GPT 해석 결과</h3>
            <p className="text-sm whitespace-pre-wrap">
              {analysisResult.gpt_interpretation}
            </p>
          </div>
        )}
      </div>

      {/* 분석 결과 영역 */}
      {analysisResult && (
        <div className="space-y-8">
          {/* 분석된 이미지 */}
          <div className="relative w-full">
            <img 
              src={analysisImage} 
              alt="Analyzed frame" 
              className="w-full h-auto rounded-lg"
            />
            
            {/* 스켈레톤 오버레이 */}
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

          {/* 분석 결과 텍스트 */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 행동 인식 결과 */}
              <div className="space-y-4">
                {/* 기존 분석 결과 컴포넌트들 */}
                {/* ... */}
              </div>

              {/* 상세 분석 및 예측 */}
              <div className="space-y-4">
                {/* 기존 상세 분석 및 예측 컴포넌트들 */}
                {/* ... */}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WebcamComponent;
