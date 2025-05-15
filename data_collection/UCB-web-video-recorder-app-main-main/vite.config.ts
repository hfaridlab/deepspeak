import { resolve } from 'path';
import react from '@vitejs/plugin-react-swc';
import fs from 'fs';
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/video/',
  plugins: [react()],
  resolve: {
    alias: {
      features: resolve(__dirname, 'src/features'),
      components: resolve(__dirname, 'src/components'),
      hooks: resolve(__dirname, 'src/hooks'),
      contexts: resolve(__dirname, 'src/contexts'),
      services: resolve(__dirname, 'src/services'),
      styles: resolve(__dirname, 'src/styles'),
      assets: resolve(__dirname, 'src/assets'),
    },
  },
});
