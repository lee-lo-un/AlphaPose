'use client';

import Image from 'next/image';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="w-full h-screen bg-white flex">
      {/* 왼쪽 영역 (로고) */}
      <div className="w-[8.33%] pl-20 pt-10">
        <div className="w-[400px]">
          <Image
            src="/images/logo_new.png"
            alt="Alpha Pose Logo"
            width={400}
            height={200}
            className="object-contain w-full h-full"
            priority
          />
        </div>
      </div>

      {/* 중앙 텍스트 영역 - 1/6 지점에 위치 */}
      <div className="w-[25%] ml-[8.33%] flex flex-col justify-center">
        <div className="flex flex-col items-center">
          <h1 className="text-[100px] leading-tight font-bold text-black text-left">
            Alpha<br />
            Pose
          </h1>
          <div className="mt-8 bg-gray-50 p-6 rounded-lg flex justify-center">
            <p className="w-[334px] h-[250px] bg-gray-200 rounded-[32px] text-[28px] font-bold text-gray-600 flex items-center justify-center leading-relaxed px-4">
              사람의 행동을 인식하고<br />
              파악하는 기능을 만들고<br />
              있습니다.<br />
              AI 연동으로 사람의<br />
              포즈를 이해합니다.
            </p>
          </div>
        </div>
      </div>

      {/* 오른쪽 영역 (버튼들) */}
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col gap-4 rotate-135">
          <div className="flex gap-6">
            {/* 파란색 버튼 */}
            <div 
              onClick={() => router.push('/photo')}
              className="w-[200px] h-[200px] bg-[#0066CC] rounded-[32px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl"
            />
            {/* 초록색 버튼 */}
            <div 
              onClick={() => router.push('/gallery')}
              className="w-[200px] h-[200px] bg-[#009966] rounded-[32px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl"
            />
          </div>
          <div className="flex gap-6">
            {/* 빨간색 버튼 */}
            <div 
              onClick={() => router.push('/live')}
              className="w-[200px] h-[200px] bg-[#CC0000] rounded-[32px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl"
            />
            {/* 회색 버튼 */}
            <div 
              className="w-[450px] h-[450px] bg-gray-200 rounded-[32px] cursor-pointer transform transition-all hover:scale-105 shadow-lg hover:shadow-xl"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
