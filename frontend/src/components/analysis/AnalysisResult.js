import { useState, useRef } from 'react';

// 분석 결과 표시를 담당하는 컴포넌트
export default function AnalysisResult({ analysisResult, analysisImage }) {
  if (!analysisResult) return null;

  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const imageRef = useRef(null);

  // 이미지가 로드되면 실제 크기를 저장
  const handleImageLoad = () => {
    if (imageRef.current) {
      setImageDimensions({
        width: imageRef.current.naturalWidth, // 원본 이미지의 실제 크기
        height: imageRef.current.naturalHeight
      });
    }
  };

  // 좌표 변환 함수
  const scaleCoordinates = (x, y) => {
    const scaleX = imageRef.current.clientWidth / imageDimensions.width;
    const scaleY = imageRef.current.clientHeight / imageDimensions.height;
    return {
      x: x * scaleX,
      y: y * scaleY
    };
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="space-y-6 max-w-3xl mx-auto">
        {/* 분석된 이미지 */}
        <div className="relative w-full">
          <img 
            ref={imageRef}
            src={analysisImage} 
            alt="Analyzed frame" 
            className="w-full h-auto rounded-lg"
            onLoad={handleImageLoad}
          />
          
          {/* 스켈톤 오버레이 */}
          {analysisResult.skeleton_data && imageDimensions.width > 0 && (
            <div className="absolute top-0 left-0 w-full h-full">
              {Object.entries(analysisResult.skeleton_data.keypoints).map(([key, kp], idx) => {
                const scaled = scaleCoordinates(kp.x, kp.y);
                return (
                  <div
                    key={`analysis-kp-${idx}`}
                    className="absolute w-2 h-2 bg-red-500 rounded-full"
                    style={{
                      left: `${scaled.x}px`,
                      top: `${scaled.y}px`,
                      opacity: kp.score > 0.5 ? 1 : 0.3,
                      transform: 'translate(-50%, -50%)'
                    }}
                  />
                );
              })}
            </div>
          )}

          {/* 객체 감지 박스 오버레이 */}
          {analysisResult.object_data && analysisResult.object_data.length > 0 && imageDimensions.width > 0 && (
            <>
              {analysisResult.object_data.map((obj, idx) => {
                const topLeft = scaleCoordinates(obj.bbox[0], obj.bbox[1]);
                const bottomRight = scaleCoordinates(obj.bbox[2], obj.bbox[3]);
                return (
                  <div
                    key={`analysis-obj-${idx}`}
                    className="absolute border-2 border-green-500"
                    style={{
                      left: `${topLeft.x}px`,
                      top: `${topLeft.y}px`,
                      width: `${bottomRight.x - topLeft.x}px`,
                      height: `${bottomRight.y - topLeft.y}px`
                    }}
                  >
                    <span className="absolute -top-6 left-0 bg-green-500 text-white px-2 py-1 text-xs rounded whitespace-nowrap">
                      {obj.class} ({obj.confidence?.toFixed(2)})
                    </span>
                  </div>
                );
              })}
            </>
          )}
        </div>

        {/* GPT 해석 결과 표시 영역 */}
        {analysisResult?.action_result?.action && (
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-4 text-gray-800">분석 결과</h3>
            <p className="text-sm whitespace-pre-wrap text-gray-700">
              {typeof analysisResult.action_result.action === 'string' 
                ? analysisResult.action_result.action 
                : JSON.stringify(analysisResult.action_result.action, null, 2)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
