/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  server: {
    port: 3000,
    host: '0.0.0.0',
  },
  // 포트 재사용 설정
  experimental: {
    allowMiddlewareResponseBody: true,
  },
}

module.exports = nextConfig 