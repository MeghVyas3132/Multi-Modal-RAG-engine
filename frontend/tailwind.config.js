/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        surface: {
          DEFAULT: '#f8f9fb',
          50: '#fafbfc',
          100: '#f1f3f5',
          200: '#e9ecef',
        },
        sidebar: {
          DEFAULT: '#111118',
          hover: '#1c1c27',
          active: '#252532',
          border: '#2a2a3a',
        },
      },
      boxShadow: {
        'soft': '0 1px 3px 0 rgba(0,0,0,0.04), 0 1px 2px -1px rgba(0,0,0,0.03)',
        'soft-md': '0 4px 12px -2px rgba(0,0,0,0.06), 0 2px 4px -2px rgba(0,0,0,0.04)',
        'soft-lg': '0 8px 24px -4px rgba(0,0,0,0.08), 0 4px 8px -4px rgba(0,0,0,0.04)',
        'glow': '0 0 0 1px rgba(0,0,0,0.03), 0 2px 8px rgba(0,0,0,0.04)',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.25rem',
      },
    },
  },
  plugins: [],
}
