/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#eff6ff',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
        },
        dark: {
          50:  '#f0f0f5',
          100: '#e0e0eb',
          400: '#8888aa',
          500: '#55556a',
          600: '#2a2a3a',
          700: '#1a1a28',
          800: '#12121e',
          900: '#0a0a0f',
        }
      }
    }
  },
  plugins: [],
}
