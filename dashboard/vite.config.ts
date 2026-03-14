import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import mdx from '@mdx-js/rollup'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import remarkFrontmatter from 'remark-frontmatter'
import remarkMdxFrontmatter from 'remark-mdx-frontmatter'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [
    mdx({
      include: /\.md$/,
      remarkPlugins: [
        remarkGfm,
        remarkMath,
        remarkFrontmatter,
        [remarkMdxFrontmatter, { name: 'frontmatter' }],
      ],
      rehypePlugins: [rehypeKatex, rehypeSlug],
    }),
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@vault': path.resolve(__dirname, '..'),
      // Force external files to use blog's node_modules
      'katex': path.resolve(__dirname, 'node_modules/katex'),
      'react-katex': path.resolve(__dirname, 'node_modules/react-katex'),
      'lucide-react': path.resolve(__dirname, 'node_modules/lucide-react'),
      'react': path.resolve(__dirname, 'node_modules/react'),
      'react-dom': path.resolve(__dirname, 'node_modules/react-dom'),
      'react-router-dom': path.resolve(__dirname, 'node_modules/react-router-dom'),
    },
  },
  build: {
    outDir: 'dist',
  },
  server: {
    host: '0.0.0.0', // Listen on all network interfaces (enables Tailscale access)
    port: 5173,
    allowedHosts: [
      'mini',
      'mini.tail-scale.ts.net',
      '.tail-scale.ts.net',
      '.trycloudflare.com',
      'madder.briankitano.com',
      'madder.netlify.app',
    ],
    fs: {
      // Allow serving files from the memory-and-attention project
      allow: [
        path.resolve(__dirname, '..'),
      ],
    },
    proxy: {
      // Proxy MAD dashboard API requests to local SSE server in development
      '/api/mad': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/mad/, ''),
      },
    },
  },
})
