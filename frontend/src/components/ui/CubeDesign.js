'use client';

import { useRouter } from 'next/navigation';
import { CameraIcon, FileIcon, VideoIcon } from '@/components/icons/DiamondIcons';
import { useState } from 'react';

export const CubeDesign = () => {
  const router = useRouter();
  const [hoverText, setHoverText] = useState('');

  return (
    <div className="absolute top-8 left-12">
      <div className="flex flex-col gap-2 rotate-135 scale-30">
        <div className="flex gap-2">
          {/* 파란색 버튼 */}
          <CubeButton
            color="#0066CC"
            icon={<CameraIcon />}
            onClick={() => router.push('/photo')}
          />
          {/* 초록색 버튼 */}
          <CubeButton
            color="#009966"
            icon={<FileIcon />}
            onClick={() => router.push('/upload')}
            onMouseEnter={() => setHoverText('')}
            onMouseLeave={() => setHoverText('')}
          />
        </div>
        <div className="flex gap-2">
          {/* 빨간색 버튼 */}
          <CubeButton
            color="#CC0000"
            icon={<VideoIcon />}
            onClick={() => router.push('/live')}
          />
          {/* 회색 버튼 (크기 커짐, Home 이동) */}
          <CubeButton
            color="#d3d3d3"
            style={{ width: '100px', height: '100px' }}
            onClick={() => router.push('/')}
            children={
              <div className="flex flex-col items-center justify-center text-center -rotate-135 font-bold text-lg">
                <span>Alpha</span>
                <span>Pose</span>
              </div>
            }
          />
        </div>
      </div>
      {/* 호버 텍스트 */}
      {hoverText && (
        <div className="absolute top-28 left-16 bg-gray-800 text-white px-2 py-1 rounded shadow">
          {hoverText}
        </div>
      )}
    </div>
  );
};

const CubeButton = ({
  color,
  icon,
  onClick,
  onMouseEnter,
  onMouseLeave,
  style = {},
  className = '',
  children = null, // 추가된 children 속성
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
      ...style,
    }}
  >
    {icon && (
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -rotate-135 scale-[0.4]">
        {icon}
      </div>
    )}
    {children && <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">{children}</div>}
  </div>
);
