import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'runtime-config-default',
      configureServer(server) {
        server.middlewares.use('/runtime-config.js', (_req, res) => {
          res.setHeader('Content-Type', 'application/javascript')
          res.setHeader('Cache-Control', 'no-store')
          res.end('/* Local development uses VITE_* defaults. */')
        })
      },
      generateBundle() {
        this.emitFile({
          type: 'asset',
          fileName: 'runtime-config.js',
          source:
            '/* Runtime configuration is injected by the container entrypoint. */',
        })
      },
    },
  ],
  server: {
    watch: {
      ignored: [
        '**/.references/**',
        '**/test-results/**',
        '**/playwright-report/**',
        '**/.tmp/**',
      ],
    },
  },
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
  build: {
    // Local runtime snapshots must never be copied into distributable assets.
    copyPublicDir: false,
    rollupOptions: {
      output: {
        manualChunks: (id: string) =>
          id.includes('echarts') || id.includes('zrender')
            ? 'charts'
            : undefined,
      },
    },
  },
})
