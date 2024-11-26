// 분석 결과 표시를 담당하는 컴포넌트
const AnalysisResult = ({ analysisResult, analysisImage }) => {
  if (!analysisResult) return null;

  return (
    <div className="space-y-8">
      <AnalyzedImage 
        image={analysisImage} 
        skeletonData={analysisResult.skeleton_data}
        objectData={analysisResult.object_data}
      />
      <GptInterpretation result={analysisResult.action_result} />
    </div>
  );
}; 