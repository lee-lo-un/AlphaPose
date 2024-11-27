'use client';

import { WebcamComponent } from '@/components/webcam';
import { useRouter } from 'next/navigation';
import { CameraIcon, FileIcon, VideoIcon } from '@/components/icons/DiamondIcons';
import { useState } from 'react';
import { CubeDesign } from '@/components/ui/CubeDesign';

export default function LivePage() {
  const router = useRouter();
  const [hoverText, setHoverText] = useState('');

  return (
    <div className="relative">
      {/* 헤더 */}
      <header className="w-full top-0 left-0 bg-white shadow-md z-15 flex items-center">
        <CubeDesign />
      </header>
  
      {/* 메인 콘텐츠 */}
      <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-32 pt-20">
        <WebcamComponent />
      </main>
    </div>
  );
}
const CubeButton = ({
  color,
  icon,
  onClick,
  onMouseEnter,
  onMouseLeave,
  style = {}, // 추가된 style 속성
  className = '', // 추가된 className 속성
}) => (
  <div
    onClick={onClick}
    onMouseEnter={onMouseEnter}
    onMouseLeave={onMouseLeave}
    className={`relative rounded-[10px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl flex items-center justify-center ${className}`}
    style={{
      backgroundColor: color,
      width: '60px',
      height: '60px',
      ...style, // style 속성 병합
    }}
  >
    {icon && (
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -rotate-135 scale-[0.4]">
        {icon}
      </div>
    )}
  </div>
);
