'use client';

import { WebcamComponent } from '@/components/webcam';

export default function LivePage() {
  console.log('LivePage rendered'); // 페이지 렌더링 확인

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      <WebcamComponent />
    </main>
  );
}