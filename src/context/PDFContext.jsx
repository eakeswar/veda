import { createContext, useContext, useEffect, useReducer } from 'react'

const PDFContext = createContext(null)

const initialState = {
  pdfDoc: null,
  toc: [],
  selectedPage: null,
  pageText: '',
  pageOffset: 0,
  ttsState: 'idle',
  language: 'en-US',
  wordIndex: -1,
  voiceProvider: 'cloud',
  voiceName: '',
  theme: 'light',
  notification: null,
}

function reducer(state, action) {
  switch (action.type) {
    case 'SET_PDF': return { ...state, pdfDoc: action.payload, toc: [], selectedPage: null, pageText: '', pageOffset: 0, ttsState: 'idle', wordIndex: -1 }
    case 'SET_TOC': return { ...state, toc: action.payload }
    case 'SET_PAGE': return { ...state, selectedPage: action.payload, wordIndex: -1 }
    case 'SET_PAGE_TEXT': return { ...state, pageText: action.payload }
    case 'SET_PAGE_OFFSET': return { ...state, pageOffset: action.payload }
    case 'SET_TTS_STATE': return { ...state, ttsState: action.payload }
    case 'SET_LANGUAGE': return { ...state, language: action.payload }
    case 'SET_WORD_INDEX': return { ...state, wordIndex: action.payload }
    case 'SET_VOICE_PROVIDER': return { ...state, voiceProvider: action.payload, ttsState: 'idle', wordIndex: -1 }
    case 'SET_VOICE_NAME': return { ...state, voiceName: action.payload }
    case 'SET_THEME': return { ...state, theme: action.payload }
    case 'SET_NOTIFICATION': return { ...state, notification: action.payload }
    case 'CLEAR_NOTIFICATION': return { ...state, notification: null }
    default: return state
  }
}

export const SARVAM_CREDITS_NOTIFICATION = {
  id: 'sarvam-credits',
  type: 'warning',
  message:
    'Your Sarvam API credits are exhausted. Add credits at dashboard.sarvam.ai to restore TTS, translation, and read-along.',
}

export function notifySarvamCreditsExhausted(dispatch) {
  dispatch({ type: 'SET_NOTIFICATION', payload: SARVAM_CREDITS_NOTIFICATION })
}

export function PDFProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', state.theme)
    document.documentElement.style.colorScheme = state.theme
  }, [state.theme])

  return (
    <PDFContext.Provider value={{ state, dispatch }}>
      <div className="h-full min-h-screen">
        {children}
      </div>
    </PDFContext.Provider>
  )
}

export function usePDF() {
  return useContext(PDFContext)
}
