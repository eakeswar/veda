import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { usePDF, notifySarvamCreditsExhausted } from '../context/PDFContext'
import { extractPagePresentation, extractPageText, fetchSemanticAnalysis, fetchTeluguDeck, fetchTranslatedFields, upscaleImages } from '../utils/pdfUtils'
import { requestCloudTTS, requestCloudTTSBoundaries, getTTSStreamUrl, supportsCloudTTS } from '../utils/speechUtils'
import { cn } from '../lib/cn'
import AIAvatar from './AIAvatar'

function splitWords(text) {
  const words = []
  const regex = /\S+/g
  let match = regex.exec(text)

  while (match) {
    words.push({
      text: match[0],
      start: match.index,
      end: match.index + match[0].length,
    })
    match = regex.exec(text)
  }

  return words
}

function alignWords(text, boundaries) {
  const words = []
  let textIdx = 0
  const normalizedText = text.toLowerCase()

  for (const wb of boundaries) {
    if (!wb.text) continue
    const wordText = wb.text.toLowerCase()
    let foundIdx = normalizedText.indexOf(wordText, textIdx)

    if (foundIdx === -1) {
      foundIdx = normalizedText.indexOf(wordText, 0)
    }

    if (foundIdx === -1) {
      foundIdx = textIdx
    }

    let endIdx = foundIdx + wordText.length
    while (endIdx < text.length && !/\s/.test(text[endIdx])) {
      endIdx++
    }

    words.push({
      text: text.slice(foundIdx, endIdx),
      start: wb.start,
      end: wb.end
    })

    textIdx = endIdx
  }

  if (words.length === 0) {
    return splitWords(text)
  }

  return words
}


function getSelectedEntry(toc, selectedPage) {
  if (!toc || toc.length === 0) return null
  const exact = toc.find((item) => item.page === selectedPage)
  if (exact) return exact

  let closest = null
  for (const item of toc) {
    if (item.page <= selectedPage) {
      if (!closest || item.page > closest.page) {
        closest = item
      }
    }
  }
  return closest
}

function getProgressPercent(wordIndex, totalWords) {
  if (!totalWords || wordIndex < 0) {
    return 0
  }

  return Math.min(100, ((wordIndex + 1) / totalWords) * 100)
}

// Jaccard word-overlap similarity between two title strings (0–1).
function titleSimilarity(a = '', b = '') {
  const words = (s) => new Set(s.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(/\s+/).filter(Boolean))
  const wa = words(a)
  const wb = words(b)
  const intersection = [...wa].filter((w) => wb.has(w)).length
  const union = new Set([...wa, ...wb]).size
  return union === 0 ? 0 : intersection / union
}

function mergeSemanticDeck(deck, semanticData) {
  if (!semanticData) return deck

  const mergedIsDigest = deck.isDigest || !!semanticData.isDigest

  // Match each LLM topic to the closest heuristic topic by title similarity so
  // images (which were spatially matched to heuristic topics) transfer correctly
  // even when the LLM reorders topics by importance.
  const mergedTopics =
    (semanticData.isDigest && semanticData.topics)
      ? semanticData.topics.map((t, i) => {
        let bestIdx = -1
        let bestScore = -1
        // Prefer title_en (English title preserved before translation) for matching
        // so Telugu topic titles can still be matched against English heuristic titles.
        const matchTitle = t.title_en || t.title
        ;(deck.topics || []).forEach((ht, idx) => {
          const score = titleSimilarity(matchTitle, ht.title)
          if (score > bestScore) { bestScore = score; bestIdx = idx }
        })
        // If no title similarity found (e.g. fully different language, score stayed 0),
        // fall back to positional matching so images stay consistent.
        if (bestScore <= 0 && i < (deck.topics || []).length) {
          bestIdx = i
        }
        return {
          ...t,
          body: t.body || t.summary || '',
          image: bestIdx >= 0 ? (deck.topics[bestIdx]?.image || null) : null,
        }
      })
      : deck.topics
  const mergedNarration =
    mergedIsDigest
      ? (semanticData.isDigest && semanticData.narration ? semanticData.narration : deck.narration)
      : semanticData.narration

  const mergedHighlights = mergedIsDigest
    ? (semanticData.isDigest && semanticData.highlights && semanticData.highlights.length > 0
      ? semanticData.highlights
      : mergedTopics.map((t) => `${t.title}: ${t.body || t.summary || ''}`))
    : semanticData.highlights

  const mergedSupportingPoints = mergedIsDigest
    ? (semanticData.isDigest && semanticData.supportingPoints && semanticData.supportingPoints.length > 0
      ? semanticData.supportingPoints
      : mergedTopics.map((t) => t.body || t.summary || ''))
    : semanticData.supportingPoints

  const mergedNarrationTe =
    mergedIsDigest
      ? (semanticData.isDigest && semanticData.narration_te ? semanticData.narration_te : deck.narration_te)
      : semanticData.narration_te

  return {
    ...deck,
    title: semanticData.title,
    subtitle: semanticData.subtitle,
    summary: semanticData.summary,
    highlights: mergedHighlights,
    supportingPoints: mergedSupportingPoints,
    narration: mergedNarration,
    narration_te: mergedNarrationTe,
    topics: mergedTopics,
    isDigest: mergedIsDigest,
    isSemantic: true
  }
}

/** Apply a {originalUrl → upscaledUrl} map to deck.images and topics[*].image. */
function applyUpscaledImages(deck, urlMap) {
  if (!deck || !urlMap || Object.keys(urlMap).length === 0) return deck
  return {
    ...deck,
    images: (deck.images || []).map((u) => urlMap[u] || u),
    topics: (deck.topics || []).map((t) => ({
      ...t,
      image: t.image ? (urlMap[t.image] || t.image) : t.image,
    })),
  }
}

function getSpeechText(deck, language, pageText, digestTopicIndex = 0) {
  if (!deck) return pageText || ''

  if (deck.isDigest) {
    const topics = deck.topics || []
    const activeTopic = topics[digestTopicIndex]
    if (!activeTopic) return ''

    if (language === 'te-IN') {
      return activeTopic.narration_te
        || `${activeTopic.title || ''}. ${activeTopic.summary || activeTopic.body || ''}`.replace(/\s+/g, ' ').trim()
    }
    const title = activeTopic.title || ''
    const body = activeTopic.summary || activeTopic.body || ''
    return `${title}. ${body}`.replace(/\s+/g, ' ').trim()
  }

  if (language === 'te-IN') {
    return deck.narration_te || deck.narration || pageText || ''
  }
  return deck.narration || pageText || ''
}


function OriginalPageModal({ isOpen, onClose, canvasRef }) {
  const [dataUrl, setDataUrl] = useState('')

  useEffect(() => {
    if (isOpen && canvasRef.current) {
      try {
        setDataUrl(canvasRef.current.toDataURL('image/png'))
      } catch (err) {
        console.error('Error generating page data URL:', err)
      }
    } else {
      setDataUrl('')
    }
  }, [isOpen, canvasRef])

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center p-8 bg-black/70 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-3xl p-5 max-w-[90vw] max-h-[90vh] flex flex-col gap-4 shadow-2xl relative"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center border-b border-veda-border pb-2">
          <span className="font-bold text-veda-text text-lg">Original PDF Page Render</span>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-full bg-black/5 border-none cursor-pointer text-[#555555] hover:bg-black/10"
          >
            ✕
          </button>
        </div>

        <div className="overflow-auto flex justify-center items-center bg-[#f5f5f5] rounded-xl p-2">
          {dataUrl ? (
            <img
              src={dataUrl}
              alt="High-resolution PDF Page"
              className="max-w-full max-h-[72vh] object-contain rounded-lg shadow-md border border-[#e0e0e0]"
            />
          ) : (
            <p className="text-veda-muted p-8">Rendering high-res page...</p>
          )}
        </div>
      </div>
    </div>
  )
}

function EnlargedImageModal({ imageUrl, title, onClose }) {
  const [renderedImageUrl, setRenderedImageUrl] = useState(imageUrl)
  const [renderedTitle, setRenderedTitle] = useState(title)
  const [isClosing, setIsClosing] = useState(false)

  useEffect(() => {
    if (imageUrl) {
      setRenderedImageUrl(imageUrl)
      setRenderedTitle(title)
      setIsClosing(false)
    } else {
      setRenderedImageUrl(null)
      setIsClosing(false)
    }
  }, [imageUrl, title])

  const handleClose = useCallback(() => {
    if (isClosing) return
    setIsClosing(true)
    setTimeout(() => {
      onClose()
    }, 300)
  }, [isClosing, onClose])

  useEffect(() => {
    if (!renderedImageUrl) return
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        handleClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [renderedImageUrl, handleClose])

  if (!renderedImageUrl) return null

  return (
    <div
      className={cn(
        'fixed inset-0 z-[2000] flex flex-col items-center justify-center p-8 cursor-default',
        isClosing ? 'animate-fade-out-backdrop' : 'animate-fade-in-backdrop'
      )}
      onClick={handleClose}
    >
      <button
        onClick={(e) => {
          e.stopPropagation()
          handleClose()
        }}
        className="absolute top-6 right-6 z-[2100] w-11 h-11 flex items-center justify-center rounded-full
          bg-white/10 border border-white/15 text-white backdrop-blur-xl shadow-lg
          transition-all duration-300 hover:bg-white/20 hover:scale-110 hover:border-white/30 hover:shadow-xl"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 1L13 13M1 13L13 1" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      <div
        className={cn(
          'flex flex-col items-center gap-4 max-w-[85vw] max-h-[80vh] cursor-default',
          isClosing ? 'animate-zoom-out-spring' : 'animate-zoom-in-spring'
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={renderedImageUrl}
          alt={renderedTitle || "Enlarged view"}
          className="max-w-full max-h-[75vh] object-contain rounded-[20px] border border-white/10 shadow-2xl"
        />
        {renderedTitle && (
          <span
            className="text-white text-sm font-medium font-display text-center mt-1.5
              bg-[rgba(20,20,20,0.65)] border border-white/10 px-5 py-2 rounded-full
              backdrop-blur-xl shadow-2xl"
          >
            {renderedTitle}
          </span>
        )}
      </div>
    </div>
  )
}

export default function RightPanel({ sidebarOpen, onToggleSidebar }) {
  const { state, dispatch } = usePDF()
  const isLight = state.theme === 'light'
  const sarvamCreditsNotifiedRef = useRef(false)
  const onSarvamCreditsExhausted = useCallback(() => {
    if (!sarvamCreditsNotifiedRef.current) {
      sarvamCreditsNotifiedRef.current = true
      notifySarvamCreditsExhausted(dispatch)
    }
  }, [dispatch])
  const renderCanvasRef = useRef(null)
  const renderTaskRef = useRef(null)
  const audioRef = useRef(null)
  const objectUrlRef = useRef(null)
  const cloudWordBoundariesRef = useRef([])
  const [cloudMessage, setCloudMessage] = useState('')
  const [pageDeck, setPageDeck] = useState(null)
  const [isPreparing, setIsPreparing] = useState(false)
  const [activeTab, setActiveTab] = useState('takeaways') // 'takeaways' or 'notes'
  const [activeBoundaries, setActiveBoundaries] = useState([])
  const [activeAudio, setActiveAudio] = useState(null)
  const [isOriginalModalOpen, setIsOriginalModalOpen] = useState(false)
  const [enlargedImage, setEnlargedImage] = useState(null)
  const [enlargedTitle, setEnlargedTitle] = useState('')

  // Cache for analysed pages
  const [pageCache, setPageCache] = useState({})
  const pageCacheRef = useRef(pageCache)
  useEffect(() => {
    pageCacheRef.current = pageCache
  }, [pageCache])

  // Clear cache when a new PDF document is uploaded
  useEffect(() => {
    setPageCache({})
  }, [state.pdfDoc])

  // Cache for preloaded audio and boundaries
  const [audioCache, setAudioCache] = useState({})
  const audioCacheRef = useRef(audioCache)
  useEffect(() => {
    audioCacheRef.current = audioCache
  }, [audioCache])

  // Clear audio cache and revoke URLs when a new PDF document is uploaded
  useEffect(() => {
    Object.values(audioCacheRef.current).forEach((item) => {
      if (item?.url && item.url.startsWith('blob:')) {
        URL.revokeObjectURL(item.url)
      }
    })
    setAudioCache({})
  }, [state.pdfDoc])

  const [speechRate, setSpeechRate] = useState('+0%')
  const [selectedVoice, setSelectedVoice] = useState('en-US-AriaNeural')
  const [activeImageIndex, setActiveImageIndex] = useState(0)
  const [digestTopicIndex, setDigestTopicIndex] = useState(0)
  const autoPlayAfterPreloadRef = useRef(false)

  const pageDeckRef = useRef(pageDeck)
  useEffect(() => {
    pageDeckRef.current = pageDeck
  }, [pageDeck])

  // Persists upscaled URL mappings across deck swaps on the same page so that
  // when the semantic/Telugu deck replaces the heuristic deck it immediately
  // inherits any already-upscaled images without another round-trip.
  const upscaledUrlMapRef = useRef({})

  // Tracks which page number the current pageDeck was built for.
  // Used to prevent reusing an old deck when navigating to a new page in Telugu mode.
  const pageDeckPageRef = useRef(null)

  const digestTopicIndexRef = useRef(digestTopicIndex)
  useEffect(() => {
    digestTopicIndexRef.current = digestTopicIndex
  }, [digestTopicIndex])

  // Reset selected voice when language changes
  useEffect(() => {
    if (state.language === 'te-IN') {
      setSelectedVoice('te-IN-ShrutiNeural')
    } else {
      setSelectedVoice('en-US-AriaNeural')
    }
  }, [state.language])

  const speechText = useMemo(
    () => getSpeechText(pageDeck, state.language, state.pageText, digestTopicIndex),
    [pageDeck, digestTopicIndex, state.language, state.pageText]
  )

  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [pendingSemanticDeck, setPendingSemanticDeck] = useState(null)
  const ttsStateRef = useRef(state.ttsState)
  const lastWordIndexRef = useRef(-1)
  const activeWordRef = useRef(null)

  const clearCloudAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }

    if (objectUrlRef.current) {
      if (objectUrlRef.current.startsWith('blob:')) {
        URL.revokeObjectURL(objectUrlRef.current)
      }
      objectUrlRef.current = null
    }
    setActiveAudio(null)
  }, [])



  const playPreloaded = useCallback(async (url, wordBoundaries) => {
    try {
      clearCloudAudio()
      dispatch({ type: 'SET_TTS_STATE', payload: 'speaking' })
      dispatch({ type: 'SET_WORD_INDEX', payload: 0 })
      lastWordIndexRef.current = 0
      setActiveBoundaries(wordBoundaries)

      cloudWordBoundariesRef.current = wordBoundaries
      objectUrlRef.current = url
      const audio = new Audio(url)
      audioRef.current = audio
      setActiveAudio(audio)

      audio.onended = () => {
        dispatch({ type: 'SET_TTS_STATE', payload: 'idle' })
        dispatch({ type: 'SET_WORD_INDEX', payload: -1 })
        lastWordIndexRef.current = -1
        setActiveBoundaries([])
        setCloudMessage('Natural voice playback complete')

        if (pageDeckRef.current?.isDigest) {
          const topics = pageDeckRef.current.topics || []
          if (digestTopicIndexRef.current < topics.length - 1) {
            autoPlayAfterPreloadRef.current = true
            setDigestTopicIndex((idx) => idx + 1)
          }
        }
      }

      audio.onpause = () => {
        if (audio.currentTime > 0 && !audio.ended) {
          dispatch({ type: 'SET_TTS_STATE', payload: 'paused' })
        }
      }

      await audio.play()
      setCloudMessage('Natural voice playback active')
    } catch (error) {
      console.error(error)
      dispatch({ type: 'SET_TTS_STATE', payload: 'idle' })
      dispatch({ type: 'SET_WORD_INDEX', payload: -1 })
      setActiveBoundaries([])
      setCloudMessage('Playback failed. Please try again.')
    }
  }, [clearCloudAudio, dispatch])

  const handleStop = useCallback(() => {
    autoPlayAfterPreloadRef.current = false
    clearCloudAudio()
    dispatch({ type: 'SET_TTS_STATE', payload: 'idle' })
    dispatch({ type: 'SET_WORD_INDEX', payload: -1 })
    lastWordIndexRef.current = -1
    setActiveBoundaries([])
    setCloudMessage('')
  }, [clearCloudAudio, dispatch])

  useEffect(() => {
    ttsStateRef.current = state.ttsState
  }, [state.ttsState])

  // Keyboard left/right arrow for page navigation
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.key === 'ArrowLeft' && state.selectedPage > 1) {
        dispatch({ type: 'SET_PAGE', payload: state.selectedPage - 1 })
      } else if (e.key === 'ArrowRight' && state.pdfDoc && state.selectedPage < state.pdfDoc.numPages) {
        dispatch({ type: 'SET_PAGE', payload: state.selectedPage + 1 })
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [dispatch, state.selectedPage, state.pdfDoc])

  useEffect(() => {
    if (state.ttsState === 'idle' && pendingSemanticDeck) {
      setPageDeck(pendingSemanticDeck)
      setPendingSemanticDeck(null)
    }
  }, [state.ttsState, pendingSemanticDeck])

  // requestAnimationFrame loop for ultra-smooth sub-20ms word highlighting sync (karaoke lyrics style)
  useEffect(() => {
    let animationFrameId

    function updateProgress() {
      const audio = audioRef.current
      if (audio && !audio.paused && !audio.ended) {
        const currentTime = audio.currentTime
        const boundaries = cloudWordBoundariesRef.current

        if (boundaries && boundaries.length) {
          let currentWordIdx = boundaries.findIndex(
            (wb) => currentTime >= wb.start && currentTime < wb.end
          )

          if (currentWordIdx === -1) {
            currentWordIdx = boundaries.findIndex((wb) => wb.start > currentTime) - 1
            if (currentWordIdx < -1) {
              currentWordIdx = boundaries.length - 1
            }
          }

          if (currentWordIdx >= 0 && currentWordIdx < boundaries.length) {
            if (currentWordIdx !== lastWordIndexRef.current) {
              lastWordIndexRef.current = currentWordIdx
              dispatch({ type: 'SET_WORD_INDEX', payload: currentWordIdx })
            }
          }
        }
      }
      animationFrameId = requestAnimationFrame(updateProgress)
    }

    if (state.ttsState === 'speaking') {
      animationFrameId = requestAnimationFrame(updateProgress)
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
      }
    }
  }, [state.ttsState, dispatch])

  // Smooth scroll active word into view within the scrollable container
  useEffect(() => {
    if (state.wordIndex !== -1 && activeWordRef.current) {
      activeWordRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      })
    }
  }, [state.wordIndex])

  const selectedEntry = useMemo(
    () => getSelectedEntry(state.toc, state.selectedPage),
    [state.toc, state.selectedPage]
  )

  const words = useMemo(() => {
    if (activeBoundaries && activeBoundaries.length) {
      return alignWords(speechText, activeBoundaries)
    }
    return splitWords(speechText)
  }, [speechText, activeBoundaries])
  const progressPercent = getProgressPercent(state.wordIndex, words.length)
  const cloudEnabled = supportsCloudTTS()

  // Reset active image index and digest topic index when pageDeck changes
  useEffect(() => {
    setActiveImageIndex(0)
    setDigestTopicIndex(0)
  }, [pageDeck])

  // Stop playback when page, language, rate, or voice changes
  useEffect(() => {
    handleStop()
  }, [state.selectedPage, state.language, speechRate, selectedVoice, handleStop])

  // Preloading & Speculative Next-Page Caching/Pre-fetching
  useEffect(() => {
    if (!cloudEnabled) return undefined

    let isCancelled = false
    const currentPage = state.selectedPage
    const nextPage = currentPage + 1

    // Helper to get narration text for a deck
    function getNarrationText(deck) {
      return getSpeechText(deck, state.language, deck?.sourceText || '', 0)
    }

    async function runPreload() {
      // 1. First, check/preload CURRENT page audio
      if (speechText) {
        const currentCached = audioCacheRef.current[currentPage]
        const isCurrentMatch =
          currentCached &&
          currentCached.text === speechText &&
          currentCached.language === state.language &&
          currentCached.voice === selectedVoice &&
          currentCached.rate === speechRate

        if (!isCurrentMatch) {
          setCloudMessage('Pre-fetching narration...')
          try {
            const result = await requestCloudTTS({
              text: speechText,
              language: state.language,
              rate: speechRate,
              voice: selectedVoice,
              onSarvamCreditsExhausted,
            })
            if (isCancelled) {
              if (result.url) URL.revokeObjectURL(result.url)
              return
            }
            setAudioCache((prev) => ({
              ...prev,
              [currentPage]: {
                url: result.url,
                wordBoundaries: result.wordBoundaries,
                text: speechText,
                language: state.language,
                voice: selectedVoice,
                rate: speechRate,
              },
            }))
            setCloudMessage('Narration pre-fetched')
            if (autoPlayAfterPreloadRef.current) {
              autoPlayAfterPreloadRef.current = false
              playPreloaded(result.url, result.wordBoundaries)
            }
          } catch (error) {
            console.error('Current page preload failed:', error)
            if (!isCancelled) {
              setCloudMessage('Narration pre-fetch failed. Will stream on play.')
            }
          }
        } else {
          setCloudMessage('Narration pre-fetched')
          if (autoPlayAfterPreloadRef.current) {
            autoPlayAfterPreloadRef.current = false
            playPreloaded(currentCached.url, currentCached.wordBoundaries)
          }
        }
      }

      // 2. Next, check if we can speculative-preload the NEXT page
      // [DISABLED TO CONSERVE CREDITS] Speculative preloading of the next page is disabled.
      /*
      if (state.pdfDoc && nextPage <= state.pdfDoc.numPages) {
        let nextDeck = pageCacheRef.current[nextPage]

        // 2a. If next page is not in pageCache, extract and analyze in background
        if (!nextDeck) {
          try {
            console.log(`[Preload] Extracting next page ${nextPage}...`)
            await state.pdfDoc.getPage(nextPage)
            if (isCancelled) return

            const tocEntry = getSelectedEntry(state.toc, nextPage)
            const deck = await extractPagePresentation(
              state.pdfDoc,
              nextPage,
              tocEntry?.title || `Page ${nextPage}`
            )
            if (isCancelled) return

            // Cache the basic deck first
            setPageCache((prev) => ({ ...prev, [nextPage]: deck }))
            nextDeck = deck

            // Trigger background semantic analysis
            console.log(`[Preload] Analysing next page ${nextPage} semantically...`)
            const semanticData = await fetchSemanticAnalysis(deck.sourceText, deck.title, deck.isDigest)
            if (isCancelled) return

            if (semanticData) {
              const mergedIsDigest = deck.isDigest || !!semanticData.isDigest
              const mergedTopics = (semanticData.isDigest && semanticData.topics)
                ? semanticData.topics.map((t, idx) => ({
                  ...t,
                  body: t.body || t.summary || '',
                  image: deck.topics[idx]?.image || null
                }))
                : deck.topics
              const mergedNarration = mergedIsDigest
                ? (semanticData.isDigest && semanticData.narration ? semanticData.narration : deck.narration)
                : semanticData.narration

              const mergedHighlights = mergedIsDigest
                ? (semanticData.isDigest && semanticData.highlights && semanticData.highlights.length > 0
                  ? semanticData.highlights
                  : mergedTopics.map((t) => `${t.title}: ${t.body || t.summary || ''}`))
                : semanticData.highlights

              const mergedSupportingPoints = mergedIsDigest
                ? (semanticData.isDigest && semanticData.supportingPoints && semanticData.supportingPoints.length > 0
                  ? semanticData.supportingPoints
                  : mergedTopics.map((t) => t.body || t.summary || ''))
                : semanticData.supportingPoints

              nextDeck = {
                ...deck,
                title: semanticData.title,
                subtitle: semanticData.subtitle,
                summary: semanticData.summary,
                highlights: mergedHighlights,
                supportingPoints: mergedSupportingPoints,
                narration: mergedNarration,
                topics: mergedTopics,
                isDigest: mergedIsDigest,
                isSemantic: true
              }
              setPageCache((prev) => ({ ...prev, [nextPage]: nextDeck }))
              console.log(`[Preload] Next page ${nextPage} analysis cached`)
            }
          } catch (err) {
            console.error('[Preload] Next page background processing failed:', err)
          }
        }

        // 2b. Now check/preload NEXT page audio
        if (nextDeck) {
          const nextSpeechText = getNarrationText(nextDeck)
          if (nextSpeechText) {
            const nextCached = audioCacheRef.current[nextPage]
            const isNextMatch =
              nextCached &&
              nextCached.text === nextSpeechText &&
              nextCached.language === state.language &&
              nextCached.voice === selectedVoice &&
              nextCached.rate === speechRate

            if (!isNextMatch) {
              try {
                console.log(`[Preload] Pre-fetching audio for next page ${nextPage}...`)
                const result = await requestCloudTTS({
                  text: nextSpeechText,
                  language: state.language,
                  rate: speechRate,
                  voice: selectedVoice,
                  onSarvamCreditsExhausted,
                })
                if (isCancelled) {
                  if (result.url) URL.revokeObjectURL(result.url)
                  return
                }
                setAudioCache((prev) => ({
                  ...prev,
                  [nextPage]: {
                    url: result.url,
                    wordBoundaries: result.wordBoundaries,
                    text: nextSpeechText,
                    language: state.language,
                    voice: selectedVoice,
                    rate: speechRate,
                  },
                }))
                console.log(`[Preload] Next page ${nextPage} audio cached`)
              } catch (error) {
                console.error('[Preload] Next page audio pre-fetch failed:', error)
              }
            }
          }
        }
      }
      */
    }

    // Wait until current page preparation and analysis are done
    if (!isPreparing && !isAnalyzing) {
      runPreload()
    }

    return () => {
      isCancelled = true
    }
  }, [
    speechText,
    state.selectedPage,
    state.language,
    speechRate,
    selectedVoice,
    state.pdfDoc,
    state.toc,
    isPreparing,
    isAnalyzing,
    playPreloaded,
    cloudEnabled
  ])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearCloudAudio()
    }
  }, [clearCloudAudio])


  useEffect(() => {
    // Always clear stale deck immediately so old page content never shows during navigation
    setPageDeck(null)
    setPendingSemanticDeck(null)
    upscaledUrlMapRef.current = {}   // reset per-page upscale cache

    if (!state.pdfDoc || !state.selectedPage) {
      setIsAnalyzing(false)
      setActiveBoundaries([])
      return undefined
    }

    // Return cached page deck instantly if analysis or Telugu translation is complete
    const cacheKey = `${state.selectedPage}_${state.language}`
    const cachedDeck = pageCacheRef.current[cacheKey]
    if (cachedDeck?.isSemantic || cachedDeck?.isTelugu) {
      handleStop()
      dispatch({ type: 'SET_PAGE_TEXT', payload: cachedDeck.sourceText || '' })
      pageDeckPageRef.current = state.selectedPage
      setPageDeck(cachedDeck)
      setIsPreparing(false)
      setIsAnalyzing(false)
      setPendingSemanticDeck(null)
      setActiveBoundaries([])
      return undefined
    }

    // Fast path: translate existing deck to Telugu (works even when local LLM is unavailable)
    if (state.language === 'te-IN') {
      const enCacheKey = `${state.selectedPage}_en-US`
      // Only fall back to pageDeckRef if it was built for THIS page (language switch, not navigation)
      const samePage = pageDeckPageRef.current === state.selectedPage
      const sourceDeck = pageCacheRef.current[enCacheKey] || (samePage ? pageDeckRef.current : null)
      if (sourceDeck && (sourceDeck.title || sourceDeck.topics?.length)) {
        let isCancelled = false
        setIsPreparing(false)
        setIsAnalyzing(true)
        setPendingSemanticDeck(null)
        setActiveBoundaries([])
        handleStop()
        pageDeckPageRef.current = state.selectedPage
        setPageDeck(sourceDeck)

        fetchTeluguDeck(sourceDeck, { onSarvamCreditsExhausted })
          .then((translatedDeck) => {
            if (isCancelled || !translatedDeck) {
              setIsAnalyzing(false)
              return
            }
            setIsAnalyzing(false)
            setPageCache((prev) => ({ ...prev, [cacheKey]: translatedDeck }))
            if (ttsStateRef.current === 'idle') {
              setPageDeck(translatedDeck)
            } else {
              setPendingSemanticDeck(translatedDeck)
            }
          })
          .catch((err) => {
            console.error('Telugu translation error:', err)
            setIsAnalyzing(false)
          })

        return () => {
          isCancelled = true
        }
      }
    }

    let isCancelled = false

    async function prepareDeck() {
      setIsPreparing(true)
      setIsAnalyzing(false)
      setPendingSemanticDeck(null)
      setActiveBoundaries([])
      handleStop()

      try {
        const page = await state.pdfDoc.getPage(state.selectedPage)
        const canvas = renderCanvasRef.current

        if (!canvas || isCancelled) {
          return
        }

        // Cancel any in-progress render task before starting a new one
        if (renderTaskRef.current) {
          renderTaskRef.current.cancel()
          renderTaskRef.current = null
        }

        const viewport = page.getViewport({ scale: 1.45 })
        canvas.width = viewport.width
        canvas.height = viewport.height

        const context = canvas.getContext('2d')
        const renderTask = page.render({ canvasContext: context, viewport })
        renderTaskRef.current = renderTask
        try {
          await renderTask.promise
        } catch (renderErr) {
          // RenderingCancelledException is expected when we cancel mid-render — ignore it
          if (renderErr?.name === 'RenderingCancelledException') return
          throw renderErr
        }
        renderTaskRef.current = null

        if (isCancelled) {
          return
        }

        const pageText = await extractPageText(state.pdfDoc, state.selectedPage)
        const deck = await extractPagePresentation(
          state.pdfDoc,
          state.selectedPage,
          selectedEntry?.title || `Page ${state.selectedPage}`
        )

        if (isCancelled) {
          return
        }

        dispatch({ type: 'SET_PAGE_TEXT', payload: pageText })

        // Helper: start upscaling for a deck's images, store results in upscaledUrlMapRef
        // so any subsequent deck swap on the same page can immediately use cached results.
        function startUpscaling(targetDeck) {
          const rawImages = [
            ...(targetDeck.images || []),
            ...(targetDeck.topics || []).map((t) => t.image).filter(Boolean),
          ]
          const toUpscale = [...new Set(rawImages.filter(Boolean))]
            .filter((u) => !upscaledUrlMapRef.current[u])   // skip already-upscaled
          if (toUpscale.length === 0) return
          upscaleImages(toUpscale)
            .then((upscaled) => {
              if (isCancelled) return
              const urlMap = Object.fromEntries(toUpscale.map((u, i) => [u, upscaled[i]]))
              // Persist in ref so future deck swaps can apply it immediately
              upscaledUrlMapRef.current = { ...upscaledUrlMapRef.current, ...urlMap }
              setPageDeck((prev) => applyUpscaledImages(prev, urlMap))
            })
            .catch(() => {/* upscaling is optional — silently ignore failures */})
        }

        // In Telugu mode, keep the skeleton up until the translated deck is ready
        // so the user never sees an English-language flash on a Telugu page.
        // In English mode, show the heuristic deck immediately for a snappy feel.
        const showImmediately = state.language !== 'te-IN'
        if (showImmediately) {
          pageDeckPageRef.current = state.selectedPage
          setPageDeck(deck)
          setIsPreparing(false)
          startUpscaling(deck)
        }

        // Kick off semantic analysis (and Telugu translation) in the background
        setIsAnalyzing(true)
        fetchSemanticAnalysis(deck.sourceText, deck.title, deck.isDigest, state.language, { onSarvamCreditsExhausted })
          .then(async (semanticData) => {
            if (isCancelled) return

            let finalDeck = null
            if (semanticData) {
              finalDeck = mergeSemanticDeck(deck, semanticData)
            } else if (state.language === 'te-IN') {
              finalDeck = await fetchTeluguDeck(deck, { onSarvamCreditsExhausted })
            }

            setIsAnalyzing(false)
            if (!finalDeck || isCancelled) return

            // Apply any already-upscaled images before displaying the final deck,
            // then kick off upscaling for any new images the LLM deck may have introduced.
            startUpscaling(finalDeck)
            finalDeck = applyUpscaledImages(finalDeck, upscaledUrlMapRef.current)

            // In Telugu mode, this is the FIRST time we show the deck (skeleton → Telugu)
            if (!showImmediately) {
              pageDeckPageRef.current = state.selectedPage
              setIsPreparing(false)
            }

            const semanticCacheKey = `${state.selectedPage}_${state.language}`
            setPageCache((prev) => ({ ...prev, [semanticCacheKey]: finalDeck }))

            if (ttsStateRef.current === 'idle') {
              setPageDeck(finalDeck)
            } else {
              setPendingSemanticDeck(finalDeck)
            }

            // Secondary fields (highlights, supportingPoints) were skipped during /analyze
            // for speed. Translate them lazily now that the slide is visible.
            if (state.language === 'te-IN') {
              fetchTranslatedFields(finalDeck, ['highlights', 'supportingPoints'], state.language)
                .then((enriched) => {
                  if (!enriched || isCancelled) return
                  setPageDeck(enriched)
                  setPageCache((prev) => ({ ...prev, [semanticCacheKey]: enriched }))
                })
                .catch(() => {})
            }
          })
          .catch((err) => {
            console.error('Background analysis error:', err)
            setIsAnalyzing(false)
            // If Telugu analysis failed and we never showed the deck, fall back to English heuristic
            if (!showImmediately && !isCancelled) {
              pageDeckPageRef.current = state.selectedPage
              setPageDeck(applyUpscaledImages(deck, upscaledUrlMapRef.current))
              setIsPreparing(false)
            }
          })
      } catch (error) {
        console.error(error)
        if (!isCancelled) {
          setIsPreparing(false)
        }
      }
    }

    prepareDeck()

    return () => {
      isCancelled = true
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel()
        renderTaskRef.current = null
      }
    }
  }, [dispatch, selectedEntry?.title, state.pdfDoc, state.selectedPage, state.language, handleStop])

  function handleLanguageChange(language) {
    dispatch({ type: 'SET_LANGUAGE', payload: language })
  }

  async function handleCloudPlay() {
    if (!speechText || !cloudEnabled) {
      return
    }

    if (state.ttsState === 'paused' && audioRef.current) {
      try {
        await audioRef.current.play()
        dispatch({ type: 'SET_TTS_STATE', payload: 'speaking' })
        setCloudMessage('Natural voice playback active')
      } catch (error) {
        console.error(error)
      }
      return
    }

    // 1. Check if the audio is in audioCache
    const cached = audioCache[state.selectedPage]
    const isCachedMatch =
      cached &&
      cached.text === speechText &&
      cached.language === state.language &&
      cached.voice === selectedVoice &&
      cached.rate === speechRate

    if (isCachedMatch) {
      console.log(`[Play] Using cached audio for page ${state.selectedPage}`)
      playPreloaded(cached.url, cached.wordBoundaries)
      return
    }

    // 2. If not cached, fall back to streaming
    try {
      setCloudMessage('Streaming natural voice...')
      dispatch({ type: 'SET_TTS_STATE', payload: 'speaking' })
      dispatch({ type: 'SET_WORD_INDEX', payload: 0 })

      const streamUrl = getTTSStreamUrl({
        text: speechText,
        language: state.language,
        rate: speechRate,
        voice: selectedVoice,
      })

      // Start playing the stream immediately (100ms start time)
      playPreloaded(streamUrl, [])

      // Fetch word boundaries in parallel to sync highlights
      requestCloudTTSBoundaries({
        text: speechText,
        language: state.language,
        rate: speechRate,
        voice: selectedVoice,
        onSarvamCreditsExhausted,
      }).then((result) => {
        const boundaries = result.wordBoundaries || []
        // Ensure that this response still applies to the active audio stream
        if (audioRef.current && audioRef.current.src === streamUrl) {
          console.log(`[Streaming] Boundaries loaded for streaming: ${boundaries.length} words`)
          cloudWordBoundariesRef.current = boundaries
          setActiveBoundaries(boundaries)
        }
      }).catch((err) => {
        console.error('Failed to load streaming boundaries:', err)
      })

    } catch (error) {
      console.error(error)
      dispatch({ type: 'SET_TTS_STATE', payload: 'idle' })
      dispatch({ type: 'SET_WORD_INDEX', payload: -1 })
      setActiveBoundaries([])
      setCloudMessage('Cloud voice is unavailable. Start the local TTS server, then try again.')
      clearCloudAudio()
    }
  }


  function handlePlay() {
    handleCloudPlay()
  }

  function handlePause() {
    if (audioRef.current && state.ttsState === 'speaking') {
      audioRef.current.pause()
      dispatch({ type: 'SET_TTS_STATE', payload: 'paused' })
    }
  }



  function handleWordClick(index) {
    const boundaries = cloudWordBoundariesRef.current
    if (audioRef.current && boundaries && boundaries[index]) {
      const time = boundaries[index].start
      audioRef.current.currentTime = time
      lastWordIndexRef.current = index
      dispatch({ type: 'SET_WORD_INDEX', payload: index })
      if (state.ttsState !== 'speaking') {
        audioRef.current.play().then(() => {
          dispatch({ type: 'SET_TTS_STATE', payload: 'speaking' })
        })
      }
    }
  }

  function handleProgressBarClick(e) {
    const bar = e.currentTarget
    const rect = bar.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const percent = Math.max(0, Math.min(100, (clickX / rect.width) * 100))
    if (audioRef.current && audioRef.current.duration) {
      const targetTime = (percent / 100) * audioRef.current.duration
      audioRef.current.currentTime = targetTime

      const boundaries = cloudWordBoundariesRef.current
      if (boundaries.length) {
        let wordIdx = boundaries.findIndex((wb) => targetTime >= wb.start && targetTime < wb.end)
        if (wordIdx === -1) {
          wordIdx = boundaries.findIndex((wb) => wb.start > targetTime) - 1
          if (wordIdx < -1) wordIdx = boundaries.length - 1
        }
        if (wordIdx >= 0 && wordIdx < boundaries.length) {
          lastWordIndexRef.current = wordIdx
          dispatch({ type: 'SET_WORD_INDEX', payload: wordIdx })
        }
      }
    }
  }

  function renderNarrationScript() {
    if (!words.length) {
      return null
    }

    const isPlayingOrPaused = state.ttsState === 'speaking' || state.ttsState === 'paused'

    return (
      <div className={cn('leading-relaxed text-[0.98rem] tracking-tight px-2 py-2 veda-text-primary', state.language === 'te-IN' ? 'lang-te text-[1.05rem]' : 'font-sans')}>
        {words.map((word, index) => {
          const isActive = state.wordIndex === index
          const isPast = state.wordIndex > index && state.wordIndex !== -1

          return (
            <span
              key={`${word.start}-${word.text}`}
              ref={isActive ? activeWordRef : null}
              onClick={() => handleWordClick(index)}
              className={cn(
                'karaoke-word',
                isActive && 'mr-0.5 px-1.5 font-bold scale-110 bg-veda-accent/15 dark:bg-white/15',
                isPast && 'font-semibold text-[#b45309] dark:text-veda-accent-dark',
                !isActive && !isPast && isPlayingOrPaused && 'text-black/30 dark:text-white/25',
                !isActive && !isPast && !isPlayingOrPaused && 'font-normal'
              )}
              style={{
                '--word-hover-color': isLight ? '#d97706' : '#ffd27a',
                textShadow: isActive && !isLight ? '0 0 15px rgba(255,255,255,0.7), 0 0 2px rgba(255,255,255,0.9)' : 'none',
              }}
            >
              {word.text}
            </span>
          )
        })}
      </div>
    )
  }


  // Accent color for SVG arrows in legacy inline spots
  const isPlayPrimary = state.ttsState !== 'speaking'
  const isPauseActive = state.ttsState === 'speaking'
  const isStopActive = state.ttsState !== 'idle'

  return (
    <div className="relative flex flex-col flex-1 h-screen overflow-hidden bg-veda-surface dark:bg-veda-surface-dark transition-colors duration-300">
      <canvas ref={renderCanvasRef} className="hidden" />

      {/* Original PDF Page modal overlay */}
      <OriginalPageModal isOpen={isOriginalModalOpen} onClose={() => setIsOriginalModalOpen(false)} canvasRef={renderCanvasRef} />

      {/* Enlarged Image modal overlay */}
      <EnlargedImageModal imageUrl={enlargedImage} title={enlargedTitle} onClose={() => setEnlargedImage(null)} />

      {/* Top Header bar */}
      <header
        className="flex items-center justify-between px-6 py-4 shrink-0 border-b transition-colors duration-300
          border-veda-border bg-gradient-to-b from-white to-veda-surface
          dark:border-veda-border-dark dark:from-veda-card-dark dark:to-[#0d0f12]"
      >
        <div className="flex items-center gap-3">
          {/* Sidebar toggle — macOS style */}
          <button
            onClick={onToggleSidebar}
            title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
            className="glass-btn w-[30px] h-[30px] rounded-lg text-xs shrink-0
              bg-black/[0.04] text-veda-muted border border-black/[0.05]
              hover:bg-black/[0.08] hover:text-veda-text
              dark:bg-white/[0.04] dark:text-veda-muted-dark dark:border-white/[0.06]
              dark:hover:bg-white/[0.08] dark:hover:text-veda-text-dark"
          >
            {sidebarOpen ? '◧' : '▣'}
          </button>

          <div className="flex flex-col gap-0.5">
            <span className="veda-text-primary text-xl font-extrabold font-display">
              {pageDeck?.title || selectedEntry?.title || 'Select a topic from the contents'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Page navigation controls */}
          {state.selectedPage && (
            <div className="veda-glass-chip">
              <button
                onClick={() => {
                  if (state.selectedPage > 1) {
                    dispatch({ type: 'SET_PAGE', payload: state.selectedPage - 1 })
                  }
                }}
                disabled={state.selectedPage <= 1}
                className="veda-icon-btn glass-btn"
                title="Previous Page"
              >
                ◀
              </button>

              <span className="veda-text-primary text-[0.78rem] font-bold min-w-[95px] text-center font-display">
                Page {state.selectedPage}
                {state.pageOffset && (state.selectedPage - state.pageOffset > 0)
                  ? ` (P. ${state.selectedPage - state.pageOffset})`
                  : ''}
                {state.pdfDoc ? ` of ${state.pdfDoc.numPages}` : ''}
              </span>

              <button
                onClick={() => {
                  if (state.pdfDoc && state.selectedPage < state.pdfDoc.numPages) {
                    dispatch({ type: 'SET_PAGE', payload: state.selectedPage + 1 })
                  }
                }}
                disabled={!state.pdfDoc || state.selectedPage >= state.pdfDoc.numPages}
                className="veda-icon-btn glass-btn"
                title="Next Page"
              >
                ▶
              </button>
            </div>
          )}

          {/* Sun/Moon Theme Switcher */}
          <button
            onClick={() => dispatch({ type: 'SET_THEME', payload: isLight ? 'dark' : 'light' })}
            className="glass-btn w-[34px] h-[34px] rounded-full text-sm
              bg-white/45 text-veda-text border border-black/5 shadow-sm
              hover:bg-white/65 hover:border-black/10
              dark:bg-white/5 dark:text-[#cfc6b1] dark:border-white/10 dark:shadow-md
              dark:hover:bg-white/10 dark:hover:border-white/20"
            title={isLight ? "Toggle Dark Mode" : "Toggle Light Mode"}
          >
            {isLight ? '🌙' : '☀️'}
          </button>
        </div>
      </header>

      {/* Main dashboard body */}
      <div className="flex-1 flex gap-6 px-6 pt-6 overflow-hidden min-h-0">
        {!state.selectedPage ? (
          <div
            className="flex flex-col items-center justify-center flex-1 gap-4 rounded-[30px] mb-6
              border border-veda-border bg-[radial-gradient(circle_at_top,rgba(217,119,6,0.04),rgba(255,255,255,0.98)_70%)]
              dark:border-veda-border-dark dark:bg-[radial-gradient(circle_at_top,rgba(245,166,35,0.08),rgba(11,12,16,0.92)_50%)]"
          >
            <div className="text-[3.4rem]">🎙️</div>
            <p className="veda-text-primary font-bold text-lg">Nothing selected yet</p>
            <p className="veda-text-muted text-sm">Pick a topic from the table of contents and Veda will present it</p>
          </div>
        ) : (
          <>
            {/* Slide Stage Area (60% width) */}
            <div className={cn('flex-[0_0_60%] flex flex-col items-stretch pb-[7.5rem] h-full min-h-0', state.language === 'te-IN' && 'lang-te')}>
              <div
                key={state.selectedPage}
                className={cn(
                  'animate-fade-in-up veda-card flex-1 w-full relative rounded-3xl p-6 flex flex-col justify-center overflow-hidden min-h-0 transition-all duration-300',
                  isAnalyzing && 'veda-card-analyzing'
                )}
              >
                {/* Status badge overlay */}
                <div className="absolute top-4 right-5 flex items-center gap-1.5">
                  {isAnalyzing && (
                    <span className="w-1.5 h-1.5 rounded-full bg-veda-accent dark:bg-veda-accent-dark animate-pulse" />
                  )}
                  <span className="text-[0.68rem] veda-text-muted font-semibold">
                    {isPreparing
                      ? 'Preparing…'
                      : isAnalyzing
                        ? 'AI refining…'
                        : pageDeck?.isSemantic
                          ? '✦ AI Enhanced'
                          : pageDeck ? 'Visual extract' : ''}
                  </span>
                </div>

                {isPreparing ? (
                  <div className="flex flex-col gap-3 w-full animate-pulse">
                    <div className="veda-skeleton h-7 w-2/3" />
                    <div className="veda-skeleton h-4 w-1/3" />
                    <div className="veda-skeleton h-3.5 w-full mt-2" />
                    <div className="veda-skeleton h-3.5 w-[88%]" />
                    <div className="veda-skeleton h-3.5 w-[72%]" />
                  </div>
                ) : pageDeck?.isDigest ? (
                  // ── Digest carousel: one story at a time ─────────────────
                  (() => {
                    const topics = pageDeck.topics || []
                    const total = topics.length
                    const activeTopic = topics[digestTopicIndex] || {}
                    // Collect all unique non-null images from the page for the gallery strip
                    const allImages = [...new Set(
                      (pageDeck.images || []).filter(Boolean).concat(
                        topics.map(t => t.image).filter(Boolean)
                      )
                    )]
                    return (
                      <div className="flex flex-col gap-2 w-full h-full min-h-0">
                        <div className="flex items-center justify-between shrink-0">
                          <div className="flex items-center gap-2">
                            <h2 className="m-0 text-[#111111] dark:text-[#fff7ea] text-lg font-extrabold tracking-tight">
                              {pageDeck.title || selectedEntry?.title}
                            </h2>
                            <span className="veda-story-count">{total} stories</span>
                          </div>
                        </div>

                        {/* Page image gallery strip — shows all extracted images, no guessed matching */}
                        {allImages.length > 0 && (
                          <div className="flex gap-2 overflow-x-auto shrink-0 pb-1 scrollbar-none">
                            {allImages.map((imgUrl, i) => (
                              <div
                                key={i}
                                onClick={() => { setEnlargedImage(imgUrl); setEnlargedTitle(pageDeck.title || selectedEntry?.title); }}
                                className="shrink-0 h-[90px] rounded-xl overflow-hidden cursor-default border border-black/8 dark:border-white/10 hover:opacity-90 transition-opacity"
                                style={{ aspectRatio: 'auto' }}
                              >
                                <img
                                  src={imgUrl}
                                  alt={`Page image ${i + 1}`}
                                  className="h-full w-auto object-cover block"
                                />
                              </div>
                            ))}
                          </div>
                        )}

                        <div
                          key={digestTopicIndex}
                          className="animate-story-fade-in flex-1 min-h-0 flex flex-col gap-2 overflow-y-auto"
                        >
                          <div className="flex flex-col gap-2 pr-1">
                            <div className="flex items-center gap-1.5 shrink-0">
                              <span className="text-[0.62rem] font-bold veda-accent-text tracking-widest uppercase">
                                Story {digestTopicIndex + 1}
                              </span>
                              <span className="w-5 h-px bg-veda-accent/40 dark:bg-veda-accent-dark/40" />
                            </div>
                            <h3 className="m-0 text-[#111111] dark:text-[#fff7ea] text-xl font-extrabold leading-tight tracking-tight">
                              {activeTopic.title}
                            </h3>
                            {(activeTopic.body || activeTopic.summary) && (
                              <p className="mt-1 mb-0 text-[#444444] dark:text-[#c8bfad] text-sm leading-relaxed">
                                {activeTopic.body || activeTopic.summary}
                              </p>
                            )}
                          </div>
                        </div>

                        <div className="animate-nav-slide-up flex items-center justify-center gap-5 shrink-0 mt-3">
                          <button
                            onClick={() => { handleStop(); setDigestTopicIndex((i) => Math.max(0, i - 1)); }}
                            disabled={digestTopicIndex === 0}
                            className={cn('glass-btn digest-nav-btn prev', digestTopicIndex === 0 ? 'veda-digest-nav-disabled' : 'veda-digest-nav-enabled')}
                          >‹</button>

                          <div className="flex gap-1.5 items-center">
                            {topics.map((_, idx) => (
                              <button
                                key={idx}
                                onClick={() => { handleStop(); setDigestTopicIndex(idx); }}
                                className={cn(
                                  'digest-dot',
                                  idx === digestTopicIndex
                                    ? 'w-5 bg-veda-accent dark:bg-veda-accent-dark border-none'
                                    : 'w-1.5 bg-black/10 dark:bg-white/15 border border-black/5 dark:border-white/10 backdrop-blur-sm'
                                )}
                              />
                            ))}
                          </div>

                          <span className="text-xs veda-text-muted font-bold min-w-10 text-center font-display">
                            {digestTopicIndex + 1} / {total}
                          </span>

                          <button
                            onClick={() => { handleStop(); setDigestTopicIndex((i) => Math.min(total - 1, i + 1)); }}
                            disabled={digestTopicIndex === total - 1}
                            className={cn('glass-btn digest-nav-btn next', digestTopicIndex === total - 1 ? 'veda-digest-nav-disabled' : 'veda-digest-nav-enabled')}
                          >›</button>
                        </div>
                      </div>
                    )
                  })()
                ) : pageDeck?.images?.length > 0 ? (
                  <div className="grid grid-cols-[1.15fr_0.85fr] gap-6 items-center w-full h-full min-h-0">
                    <div className="flex flex-col gap-2 overflow-y-auto max-h-full pr-1.5">
                      <h2 className="m-0 text-[#111111] dark:text-[#fff7ea] text-[1.65rem] leading-tight font-extrabold tracking-tight">
                        {pageDeck?.title || selectedEntry?.title}
                      </h2>
                      {pageDeck?.subtitle && (
                        <p className="veda-accent-text text-[0.94rem] leading-snug m-0 font-semibold">
                          {pageDeck.subtitle}
                        </p>
                      )}
                      {pageDeck?.summary && (
                        <p className="text-[#444444] dark:text-[#ccc2b2] text-sm leading-relaxed mt-1.5 mb-0">
                          {pageDeck.summary}
                        </p>
                      )}
                    </div>

                    <div className="flex flex-col gap-2.5 items-center h-full justify-center min-h-0">
                      <div
                        onClick={() => { setEnlargedImage(pageDeck.images[activeImageIndex]); setEnlargedTitle(pageDeck.title || selectedEntry?.title); }}
                        className="veda-slide-image"
                      >
                        <img
                          src={pageDeck.images[activeImageIndex]}
                          alt="Extracted illustration"
                          className="w-full h-full object-contain block"
                        />
                      </div>

                      {pageDeck.images.length > 1 && (
                        <div className="flex gap-1.5 justify-center">
                          {pageDeck.images.map((img, idx) => (
                            <button
                              key={idx}
                              onClick={() => setActiveImageIndex(idx)}
                              className={cn(
                                'w-7 h-7 rounded-md overflow-hidden p-0 cursor-pointer bg-transparent transition-all duration-200',
                                activeImageIndex === idx
                                  ? 'border-2 border-veda-accent dark:border-veda-accent-dark'
                                  : 'border border-veda-border dark:border-white/10'
                              )}
                            >
                              <img src={img} alt="" className="w-full h-full object-cover" />
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col gap-2.5 max-w-[85%] overflow-y-auto max-h-full pr-1.5">
                    <h2 className="m-0 text-[#111111] dark:text-[#fff7ea] text-[1.9rem] leading-tight font-extrabold tracking-tight">
                      {pageDeck?.title || selectedEntry?.title}
                    </h2>
                    {pageDeck?.subtitle && (
                      <p className="veda-accent-text text-lg leading-snug m-0 font-semibold">
                        {pageDeck.subtitle}
                      </p>
                    )}
                    {pageDeck?.summary && (
                      <p className="text-[#444444] dark:text-[#e6dccb] text-[0.94rem] leading-relaxed mt-2.5 mb-0">
                        {pageDeck.summary}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Interaction Sidebar (40% width) */}
            <div
              key={state.selectedPage}
              className={cn('animate-slide-in-right flex-[0_0_40%] flex flex-col gap-5 h-full pb-[7.5rem] min-h-0', state.language === 'te-IN' && 'lang-te')}
            >
              {/* Tabbed Insights Card */}
              <div className="veda-card flex-1 flex flex-col rounded-3xl overflow-hidden min-h-0 transition-all duration-300">
                {/* Tab Header Selector */}
                <div className="veda-tab-shell">
                  {[
                    { id: 'takeaways', label: 'TAKEAWAYS' },
                    { id: 'notes', label: 'NOTES' },
                    { id: 'source', label: 'SOURCE' },
                  ].map((tab) => {
                    const isActive = activeTab === tab.id
                    return (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={cn('veda-tab-btn', isActive ? 'veda-tab-btn-active' : 'veda-tab-btn-idle')}
                      >
                        {tab.label}
                      </button>
                    )
                  })}
                </div>

                {/* Tab Content Body */}
                <div className="p-4 flex-1 overflow-y-auto min-h-0">
                  {activeTab === 'takeaways' ? (
                    <div className="flex flex-col gap-2.5">
                      {(!pageDeck?.highlights || pageDeck.highlights.length === 0) ? (
                        <p className="veda-text-muted text-sm m-0">No takeaways generated for this page.</p>
                      ) : (
                        pageDeck.highlights.map((highlight, index) => (
                          <div key={`${highlight}-${index}`} className="veda-takeaway-item">
                            {highlight}
                          </div>
                        ))
                      )}
                    </div>
                  ) : activeTab === 'notes' ? (
                    <div className="flex flex-col gap-3">
                      {(!pageDeck?.supportingPoints || pageDeck.supportingPoints.length === 0) ? (
                        <p className="veda-text-muted text-sm m-0">No detailed notes found for this page.</p>
                      ) : (
                        pageDeck.supportingPoints.map((point, index) => (
                          <div key={`${point}-${index}`} className="flex gap-3 items-start">
                            <span className="veda-accent-text font-bold text-[0.86rem] pt-0.5">
                              {String(index + 1).padStart(2, '0')}
                            </span>
                            <span className="text-[#444444] dark:text-[#d7d1c2] text-[0.86rem] leading-relaxed">
                              {point}
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  ) : (
                    <div className="text-[#444444] dark:text-[#cfc8ba] text-sm leading-relaxed p-1">
                      {pageDeck?.sourceText || 'No extractable text found on this page.'}
                    </div>
                  )}
                </div>
              </div>

              {/* Live Read-Along Narration Card */}
              <div className="veda-studio-panel flex-1 flex flex-col p-5 transition-all duration-300 overflow-hidden min-h-0">
                <div className="flex items-center justify-between border-b border-veda-border dark:border-veda-border-dark pb-2.5 mb-3 shrink-0">
                  <div className="flex items-center gap-2">
                    {isPauseActive && (
                      <span className="w-1.5 h-1.5 rounded-full bg-veda-accent dark:bg-veda-accent-dark animate-pulse shrink-0" />
                    )}
                    <span className="veda-accent-text text-[0.72rem] tracking-widest uppercase font-bold">
                      {isPauseActive ? 'Reading' : 'Narration'}
                    </span>
                  </div>
                  <span className="text-[0.72rem] text-[#555555] dark:text-[#9d9586] bg-black/[0.04] dark:bg-white/[0.04] px-2 py-0.5 rounded-md">
                    {state.wordIndex >= 0 ? `${state.wordIndex + 1}/${words.length}` : `${words.length} words`}
                  </span>
                </div>

                <div className="flex-1 rounded-2xl border border-veda-border dark:border-white/[0.04] bg-black/[0.01] dark:bg-white/[0.005] p-3 overflow-y-auto min-h-0 transition-all duration-300">
                  {renderNarrationScript()}
                </div>
              </div>

            </div>
          </>
        )}
      </div>

      {/* Apple-style Unified Persistent Bottom Media Player Dock */}
      {state.selectedPage && (
        <div className="veda-dock">
          {/* Left section: Talking Avatar & Narration Status */}
          <div className="flex items-center gap-4 w-[30%] min-w-[240px]">
            <AIAvatar audioElement={activeAudio} state={{ ...state, isAnalyzing }} />
            <div className="flex flex-col gap-0.5">
              <span className="veda-text-muted text-[0.74rem] leading-snug font-semibold">
                {cloudMessage || 'Explanation server active'}
              </span>
            </div>
          </div>

          {/* Center section: Media controls & Progress slider */}
          <div className="flex flex-col items-center flex-1 max-w-[480px]">
            <div className="flex items-center gap-3">
              <div className="relative">
                {isPlayPrimary && speechText && !isPreparing && (
                  <span className="absolute inset-0 rounded-full veda-play-pulse pointer-events-none" />
                )}
                <button
                  onClick={handlePlay}
                  disabled={!speechText}
                  className={cn('veda-media-btn', isPlayPrimary ? 'veda-media-btn-play' : 'veda-media-btn-play veda-media-btn-play-idle')}
                  title="Play narration"
                >
                  ▶
                </button>
              </div>

              <button
                onClick={handlePause}
                disabled={!speechText || state.ttsState !== 'speaking'}
                className={cn('veda-media-btn veda-media-btn-round', isPauseActive ? 'veda-media-btn-pause-active' : 'veda-media-btn-muted')}
                title="Pause narration"
              >
                ⏸
              </button>

              <button
                onClick={handleStop}
                disabled={!speechText || state.ttsState === 'idle'}
                className={cn('veda-media-btn veda-media-btn-round', isStopActive ? 'veda-media-btn-stop-active' : 'veda-media-btn-muted')}
                title="Stop narration"
              >
                ⏹
              </button>
            </div>

            {/* Progress Slider */}
            <div className="flex items-center gap-3 w-full mt-1.5">
              <span className="text-[0.68rem] veda-text-muted font-bold min-w-7 text-right">
                {state.wordIndex >= 0 ? `${state.wordIndex + 1}` : '0'}
              </span>
              <div
                onClick={handleProgressBarClick}
                className="progress-bar-container flex-1 h-1.5 rounded-full overflow-hidden cursor-pointer relative bg-black/5 dark:bg-white/10"
              >
                <div
                  className="h-full transition-[width] duration-[140ms] linear rounded-full bg-gradient-to-r from-veda-accent to-veda-accent-dark dark:from-veda-accent-dark dark:to-[#ffd27a] veda-progress-fill"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <span className="text-[0.68rem] veda-text-muted font-bold min-w-7">
                {words.length}
              </span>
            </div>
          </div>

          {/* Right section: View page, Language, & Speed Rate */}
          <div className="flex items-center gap-3 justify-end w-[35%] min-w-[385px]">
            <div className="flex items-center gap-1.5">
              <span className="text-veda-muted-dark text-[0.65rem] font-bold">VOICE:</span>
              <select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="veda-select"
              >
                {state.language === 'te-IN' ? (
                  <>
                    <option value="te-IN-ShrutiNeural">Shruti (F)</option>
                    <option value="te-IN-MohanNeural">Mohan (M)</option>
                  </>
                ) : (
                  <>
                    <option value="en-US-AriaNeural">Aria (F)</option>
                    <option value="en-US-JennyNeural">Jenny (F)</option>
                    <option value="en-US-GuyNeural">Guy (M)</option>
                    <option value="en-IN-NeerjaNeural">Neerja (F-IN)</option>
                  </>
                )}
              </select>
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-veda-muted-dark text-[0.65rem] font-bold">RATE:</span>
              <select
                value={speechRate}
                onChange={(e) => setSpeechRate(e.target.value)}
                className="veda-select"
              >
                <option value="+0%">1.0x</option>
                <option value="+15%">1.2x</option>
                <option value="+30%">1.5x</option>
              </select>
            </div>

            {[
              { label: 'EN', value: 'en-US' },
              { label: 'TE', value: 'te-IN' },
            ].map((language) => {
              const isActive = state.language === language.value
              return (
                <button
                  key={language.value}
                  onClick={() => handleLanguageChange(language.value)}
                  className={cn('veda-lang-btn', isActive ? 'veda-lang-btn-active' : 'veda-lang-btn-idle')}
                >
                  {language.label}
                </button>
              )
            })}

            {/* High-resolution PDF view toggle */}
            <button
              onClick={() => setIsOriginalModalOpen(true)}
              className="glass-btn text-[0.7rem] px-3 py-1.5 rounded-full
                bg-black/[0.03] text-veda-text border border-black/5 shadow-sm
                dark:bg-white/[0.06] dark:text-veda-text-dark dark:border-white/10 dark:shadow-md"
              title="View original page layout"
            >
              📄 Original Page
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
