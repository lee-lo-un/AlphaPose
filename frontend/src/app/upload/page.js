'use client';

import { useState } from 'react';
import ImageUploader from '@/components/ui/ImageUploader';
import AnalysisInput from '@/components/analysis/AnalysisInput';
import AnalysisResult from '@/components/analysis/AnalysisResult';
import { CubeDesign } from '@/components/ui/CubeDesign';
export default function PhotoPage() {
  const [inputText, setInputText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisImage, setAnalysisImage] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleImageUpload = (file, preview) => {
    setUploadedFile(file);
    setAnalysisImage(preview);
    setAnalysisResult(null);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      alert('이미지를 먼저 업로드해주세요.');
      return;
    }

    setIsAnalyzing(true);
    try {
      // 이미지를 Base64로 변환
      const base64Image = analysisImage.split(',')[1];
      
      // API 요청 데이터 준비
      const requestData = {
        image: analysisImage,  // 전체 base64 문자열 전송
        text: inputText || null,
        top5_predictions: null
      };

      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`API 요청 실패: ${response.status} ${errorData}`);
      }

      const result = await response.json();
      
      // 결과 형식 맞추기
      setAnalysisResult({
        skeleton_data: result.skeleton_data,
        object_data: result.object_data,
        action_result: {
          // action_result가 객체인 경우를 처리
          action: typeof result.action_result === 'object' 
            ? result.action_result.action || JSON.stringify(result.action_result)
            : result.action_result
        }
      });

    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`분석 중 오류가 발생했습니다: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div>
      <CubeDesign />
      <div className="container mx-auto p-4 space-y-6">
        <ImageUploader onImageUpload={handleImageUpload} />
        
        <AnalysisInput 
          inputText={inputText}
          setInputText={setInputText}
          isAnalyzing={isAnalyzing}
          onAnalyze={handleAnalyze}
        />
        
        {/* 분석 결과가 있을 때만 결과 컴포넌트 표시 */}
        {analysisResult && (
          <AnalysisResult 
            analysisResult={analysisResult}
            analysisImage={analysisImage}
          />
        )}
      </div>
    </div>
  );
}

