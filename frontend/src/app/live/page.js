'use client';

import { WebcamComponent } from '@/components/webcam';

export default function LivePage() {
  console.log('LivePage rendered');

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      <WebcamComponent />
    </main>
  );
}