import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: process.env.HOST || 'localhost',
    port: parseInt(process.env.PORT || '9200', 10),
    proxy: {
      '/api': 'http://localhost:9100'
    }
  }
});
