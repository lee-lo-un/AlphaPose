'use client';

import { WebcamComponent } from '@/components/webcam';
import { useRouter } from 'next/navigation';
import { CameraIcon, FileIcon, VideoIcon } from '@/components/icons/DiamondIcons';

export default function LivePage() {
  console.log('LivePage rendered'); // 페이지 렌더링 확인
  const router = useRouter();

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      {/* 좌상단 큐브 디자인 */}
      <div className="absolute top-8 left-12">
        <div className="flex flex-col gap-2 rotate-135 scale-30">
          <div className="flex gap-2">


             {/* 파란색 버튼 */}
             <div 
                onClick={() => router.push('/photo')}
                className="relative w-[60px] h-[60px] bg-[#0066CC] rounded-[10px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl flex items-center justify-center"
              >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -rotate-135 scale-[0.4]">
                  <CameraIcon />
                </div>
              </div>
             {/* 초록색 버튼 */}
              <div 
                onClick={() => router.push('/upload')}
                onMouseEnter={() => setHoverText('이미지를 업로드하여 분석합니다')}
                onMouseLeave={() => setHoverText('')}
                className="relative w-[60px] h-[60px] bg-[#009966] rounded-[10px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl flex items-center justify-center"
              >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -rotate-135 scale-[0.4]">
                  <FileIcon />
                </div>
              </div>
                   </div>
          <div className="flex gap-2">
             {/* 빨간색 버튼 */}
              <div 
                onClick={() => router.push('/live')}
                className="relative w-[60px] h-[60px] bg-[#CC0000] rounded-[10px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl flex items-center justify-center duration-30000"
              >
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -rotate-135 scale-[0.4]">
                  <VideoIcon />
              </div>
            </div>
          {/* 회색 버튼 */}
            <div 
               className="relative w-[100px] h-[100px] bg-gray-200 rounded-[10px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl flex items-center justify-center"
             >
      

</div>
                 </div>
        </div>
      </div>


      <WebcamComponent />
    </main>
  );
}