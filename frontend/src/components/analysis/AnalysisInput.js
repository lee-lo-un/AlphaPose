// 텍스트 입력과 분석 버튼을 담당하는 컴포넌트
const AnalysisInput = ({ inputText, setInputText, onAnalyze, isAnalyzing }) => {
  return (
    <div className="w-full">
      <div className="flex gap-4">
        <input
          type="text"
          className="flex-1 p-2 border rounded-lg"
          placeholder="메시지를 입력하세요..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <AnalysisButton onClick={onAnalyze} isAnalyzing={isAnalyzing} />
      </div>
    </div>
  );
}; 