'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export default function ImageUploader({ onImageUpload }) {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        onImageUpload(file, reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, [onImageUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    multiple: false
  });

  return (
    <div className="max-w-3xl mx-auto mb-8">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-blue-400'
          }`}
      >
        <input {...getInputProps()} />
        
        {preview ? (
          <div className="relative">
            <img 
              src={preview} 
              alt="Preview" 
              className="max-h-[400px] mx-auto rounded-lg"
            />
            <p className="mt-2 text-sm text-gray-500">
              다른 이미지를 올리려면 클릭하거나 드래그하세요
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="text-4xl text-gray-400">📸</div>
            <p className="text-lg font-medium text-gray-600">
              이미지를 드래그하거나 클릭하여 업로드하세요
            </p>
            <p className="text-sm text-gray-500">
              (JPG, PNG, GIF 파일 지원)
            </p>
          </div>
        )}
      </div>
    </div>
  );
} 