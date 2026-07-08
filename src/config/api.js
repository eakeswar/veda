const DEFAULT_BASE = 'http://127.0.0.1:8765'

function normalizeBase(url) {
  return (url || DEFAULT_BASE).replace(/\/$/, '')
}

const API_BASE = normalizeBase(
  import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_LAYOUT_API_URL,
)

export const API = {
  base: API_BASE,
  upload: import.meta.env.VITE_UPLOAD_API_URL || `${API_BASE}/upload_pdf`,
  analyze: import.meta.env.VITE_ANALYZE_API_URL || `${API_BASE}/analyze`,
  tts: import.meta.env.VITE_TTS_API_URL || `${API_BASE}/tts`,
  pageLayout: (page) => `${API_BASE}/page_layout?page=${page}`,
  translateDeck: `${API_BASE}/translate_deck`,
  translateFields: `${API_BASE}/translate_fields`,
  upscale: `${API_BASE}/upscale_image`,
}

/** Headers for authenticated backend calls when VITE_VEDA_API_KEY is set. */
export function apiHeaders(extra = {}) {
  const headers = { ...extra }
  const key = import.meta.env.VITE_VEDA_API_KEY
  if (key) {
    headers.Authorization = `Bearer ${key}`
  }
  return headers
}

export default API
