export default function AnalysisInput({ inputText, setInputText, isAnalyzing, onAnalyze }) {
  return (
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
          onClick={onAnalyze}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? '분석 중...' : '분석'}
        </button>
      </div>
    </div>
  );
} 