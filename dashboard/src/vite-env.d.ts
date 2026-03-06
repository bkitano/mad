/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MAD_SSE_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
