'use client';

import Image from 'next/image';
import { useRouter } from 'next/navigation';

export default function PhotoPage() {
  const router = useRouter();

  return (
    <div className="w-full h-screen bg-white flex">

      {/* 왼쪽 영역: 이미지 */}
      
      <div className="w-1/2 flex items-center justify-center">
        <div className="relative w-[795px] h-[755px]">
          <Image
            src="/images/page_2nd.jpg"
            alt="Taking a Seat"
            fill
            className="object-contain"
            priority
          />
        </div>
      </div>

      {/* 오른쪽 영역: 텍스트와 버튼 */}
      <div className="w-1/2 flex flex-col items-start justify-center px-20 gap-8">
        <h1 className="text-5xl font-bold">
          Alpha Pose
        </h1>
        <p className="text-gray-600 text-xl max-w-[600px] leading-relaxed">
          사람의 행동을 인식하고 파악하는 기능을 만들고 있습니다. 여러 
          명 분석은 데이터가 쌓일 수록 정교한 분석 행동을 연결하고 
          단순한 행동 인식을 넘어 상황적 포즈를 학습하고 이해하는 시스템입니다.
        </p>
        <div className="flex flex-col gap-4">

            {/* 실행 버튼 */}
            <button
              onClick={() => router.push('/photo')}
              className="px-[60px] py-3 bg-red-500 text-white rounded-[20px] text-xl font-semibold hover:bg-red-600 transition-colors shadow-[0_4px_6px_rgba(0,0,0,0.3)] p-2 border-4 border-white hover:shadow-[0_25px_55px_rgba(0,0,0,0.6)]"
            >
              실행
            </button>
        </div>
      </div>
    </div>
  );
}