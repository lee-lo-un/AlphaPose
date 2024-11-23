'use client';

import Image from 'next/image';
import { useRouter } from 'next/navigation';
import NavButton from '@/components/ui/NavButton';

export default function HomePage() {
  const router = useRouter();

  return (
    <main className="min-h-screen relative">
      {/* 배경 이미지 */}
      <div className="absolute inset-0">
        <Image
          src="/images/hero-bg.png"
          alt="background"
          fill
          className="object-cover"
          priority
        />
      </div>

      {/* 메인 콘텐츠 */}
      <div className="relative z-10 flex min-h-screen">
        {/* 왼쪽: 메인 이미지와 로고 */}
        <div className="flex-1 flex items-center justify-center">
          <div className="relative">
            <Image
              src="/images/main.png"
              alt="main"
              width={600}
              height={600}
              className="object-contain"
            />
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <Image
                src="/images/logo.png"
                alt="logo"
                width={200}
                height={200}
                className="object-contain"
              />
            </div>
          </div>
        </div>

        {/* 오른쪽: 네비게이션 버튼 */}
        <div className="w-64 flex flex-col justify-center gap-6 p-8">
          <NavButton 
            onClick={() => router.push('/photo')}
            bgColor="bg-blue-500"
            label="사진 찍기"
          />
          <NavButton 
            onClick={() => router.push('/upload')}
            bgColor="bg-green-500"
            label="이미지 올리기"
          />
          <NavButton 
            onClick={() => router.push('/live')}
            bgColor="bg-red-500"
            label="실시간 영상"
          />
        </div>
      </div>
    </main>
  );
}