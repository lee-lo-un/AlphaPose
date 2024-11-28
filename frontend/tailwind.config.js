/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      spacing: {
        '20': '5rem', // 기존 기본값 (20은 5rem, 약 80px)
        '22': '5.5rem', // 88px
        '24': '6rem', // 96px
      },
      zIndex: {
        '5': '5',
        '15': '15',
      },

      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      rotate: {
        '135': '135deg',
      },
    },
  },
  plugins: [],
};
