'use client';

import { useState } from 'react';

export const TextInputComponent = () => {
  const [inputText, setInputText] = useState('');

  const handleTextSubmit = async () => {
    if (!inputText.trim()) return;

    try {
      const response = await fetch('http://localhost:8000/process_text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log('서버 응답:', data);
      setInputText('');  // 입력 필드 초기화
    } catch (error) {
      console.error('텍스트 전송 중 오류:', error);
    }
  };

  return (
    <div className="w-full max-w-4xl mt-4 p-4 bg-white rounded-lg shadow-md">
      <div className="flex gap-2">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="메시지를 입력하세요..."
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleTextSubmit();
            }
          }}
        />
        <button
          onClick={handleTextSubmit}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          전송
        </button>
      </div>
    </div>
  );
}; 