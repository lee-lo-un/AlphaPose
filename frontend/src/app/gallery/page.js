'use client';

import Image from 'next/image';
import { useState } from 'react';

export default function GalleryPage() {
  // 각 이미지별 hover 상태 관리
  const [hoverStates, setHoverStates] = useState({
    sitting: false,
    riding: false,
    hugging: false,
    looking: false,
    reading: false,
    running: false
  });

  // hover 상태 변경 함수
  const handleHover = (key, isHovered) => {
    setHoverStates(prev => ({
      ...prev,
      [key]: isHovered
    }));
  };

  return (
    <div className="w-[1920px] h-[1000px] bg-white p-16">
      {/* 제목 섹션 */}
      <h1 className="text-4xl font-bold mb-12 border-b-4 border-black inline-block pb-2">
        Our Results
      </h1>

      <div className="flex gap-8">
        {/* 왼쪽 영역: 앉다, 타다, 안다 */}
        <div className="flex-1 grid grid-cols-2 gap-8">
          {/* 앉다 */}
          <div 
            className="relative col-span-1 row-span-1 h-[300px] rounded-3xl overflow-hidden group"
            onMouseEnter={() => handleHover('sitting', true)}
            onMouseLeave={() => handleHover('sitting', false)}
          >
            <Image
              src={hoverStates.sitting ? '/images/sitting_result.jpg' : '/images/sitting.png'}
              alt="앉다"
              fill
              className="object-cover transition-all duration-300"
            />
            <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center">
              <span className="text-white text-4xl font-bold">앉다</span>
            </div>
          </div>

          {/* 안다 */}
          <div 
            className="relative col-span-1 row-span-2 h-[630px] rounded-3xl overflow-hidden group"
            onMouseEnter={() => handleHover('hugging', true)}
            onMouseLeave={() => handleHover('hugging', false)}
          >
            <Image
              src={hoverStates.hugging ? '/images/hugging_result.jpg' : '/images/hugging.png'}
              alt="안다"
              fill
              className="object-cover transition-all duration-300"
            />
            <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center">
              <span className="text-white text-4xl font-bold">안다</span>
            </div>
          </div>

          {/* 타다 */}
          <div 
            className="relative col-span-1 row-span-1 h-[300px] rounded-3xl overflow-hidden group"
            onMouseEnter={() => handleHover('riding', true)}
            onMouseLeave={() => handleHover('riding', false)}
          >
            <Image
              src={hoverStates.riding ? '/images/riding_result.jpg' : '/images/riding.png'}
              alt="타다"
              fill
              className="object-cover transition-all duration-300"
            />
            <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center">
              <span className="text-white text-4xl font-bold">타다</span>
            </div>
          </div>


        </div>

        {/* 오른쪽 영역 */}
        <div className="flex-1 grid grid-cols-2 gap-8">
          {/* 보다 */}
          <div 
            className="relative col-span-1 row-span-1 h-[300px] rounded-3xl overflow-hidden group"
            onMouseEnter={() => handleHover('looking', true)}
            onMouseLeave={() => handleHover('looking', false)}
          >
            <Image
              src={hoverStates.looking ? '/images/looking_result.jpg' : '/images/looking.png'}
              alt="보다"
              fill
              className="object-cover transition-all duration-300"
            />
            <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center">
              <span className="text-white text-4xl font-bold">보다</span>
            </div>
          </div>

          {/* 책을 읽다 */}
          <div 
            className="relative col-span-1 row-span-2 h-[630px] rounded-3xl overflow-hidden group"
            onMouseEnter={() => handleHover('reading', true)}
            onMouseLeave={() => handleHover('reading', false)}
          >
            <Image
              src={hoverStates.reading ? '/images/reading_result.jpg' : '/images/reading.png'}
              alt="책을 읽다"
              fill
              className="object-cover transition-all duration-300"
            />
            <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center">
              <span className="text-white text-4xl font-bold">책을 읽다</span>
            </div>
          </div>

          {/* 달리다 */}
          <div 
            className="relative col-span-1 row-span-1 h-[300px] rounded-3xl overflow-hidden group"
            onMouseEnter={() => handleHover('running', true)}
            onMouseLeave={() => handleHover('running', false)}
          >
            <Image
              src={hoverStates.running ? '/images/running_result.jpg' : '/images/running.png'}
              alt="달리다"
              fill
              className="object-cover transition-all duration-300"
            />
            <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center">
              <span className="text-white text-4xl font-bold">달리다</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 