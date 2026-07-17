import { useState } from "react"
import { usePDF } from "../context/PDFContext"
import { clearPageLayoutCache, extractTOC } from "../utils/pdfUtils"
import { cn } from "../lib/cn"
import API, { apiHeaders } from "../config/api"

function getActiveItem(toc, selectedPage) {
  if (!toc || toc.length === 0) return null
  const exact = toc.find((item) => item.page === selectedPage)
  if (exact) return exact
  let closest = null
  for (const item of toc) {
    if (item.page <= selectedPage && (!closest || item.page > closest.page)) {
      closest = item
    }
  }
  return closest
}

export default function LeftPanel({ isOpen, onToggle }) {
  const { state, dispatch } = usePDF()
  const [isUploading, setIsUploading] = useState(false)

  async function handleFileChange(e) {
    const file = e.target.files[0]
    if (!file) return
    setIsUploading(true)
    try {
      const pdfjsLib = await import("pdfjs-dist")
      pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
        "pdfjs-dist/build/pdf.worker.min.mjs",
        import.meta.url
      ).toString()
      const arrayBuffer = await file.arrayBuffer()
      try {
        const UPLOAD_URL = API.upload
        await fetch(UPLOAD_URL, {
          method: 'POST',
          headers: apiHeaders({ 'Content-Type': 'application/pdf' }),
          body: arrayBuffer,
        })
        console.log('PDF successfully uploaded to backend for layout analysis.')
      } catch (uploadErr) {
        console.warn('Backend PDF upload unavailable:', uploadErr)
      }
      const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise
      clearPageLayoutCache()
      dispatch({ type: "SET_PDF", payload: pdfDoc })
      const { toc, offset } = await extractTOC(pdfDoc)
      dispatch({ type: "SET_TOC", payload: toc })
      dispatch({ type: "SET_PAGE_OFFSET", payload: offset })
    } catch (err) {
      console.error("Error:", err)
    } finally {
      setIsUploading(false)
    }
  }

  const activeItem = getActiveItem(state.toc, state.selectedPage)
  const pageCount = state.pdfDoc?.numPages ?? null

  return (
    <aside
      className={cn(
        'flex flex-col h-screen shrink-0 border-r transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)] overflow-hidden',
        'bg-white border-veda-border dark:bg-[#1a1a1a] dark:border-[#2a2a2a]',
        isOpen ? 'w-[300px] opacity-100' : 'w-0 opacity-0 border-r-0'
      )}
    >
      {/* Veda App Identity Header */}
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-veda-border dark:border-[#2a2a2a] shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-veda-accent dark:bg-veda-accent-dark flex items-center justify-center shadow-sm shadow-veda-accent/30">
            <span className="text-white text-[0.6rem] font-black leading-none">V</span>
          </div>
          <span className="font-display font-black text-[0.9rem] text-veda-text dark:text-veda-text-dark tracking-tight">
            Veda
          </span>
        </div>
        {pageCount && (
          <span className="text-[0.68rem] font-semibold text-veda-muted dark:text-veda-muted-dark
            bg-black/[0.04] dark:bg-white/[0.04] px-2 py-0.5 rounded-md">
            {pageCount} pg
          </span>
        )}
      </div>

      {/* Upload button */}
      <div className="p-3 shrink-0">
        <label
          className={cn(
            'flex items-center justify-center gap-2 w-full py-2.5 px-3 rounded-xl text-center text-[0.8rem] font-semibold cursor-pointer transition-all duration-200',
            isUploading
              ? 'bg-veda-accent/70 text-white/80 cursor-not-allowed pointer-events-none'
              : 'bg-veda-accent text-white shadow-md shadow-veda-accent/25 hover:brightness-105 active:scale-[0.98] dark:bg-veda-accent-dark dark:text-veda-surface-dark dark:shadow-none'
          )}
        >
          {isUploading ? (
            <>
              <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
              </svg>
              Loading…
            </>
          ) : (
            'Upload PDF'
          )}
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            disabled={isUploading}
            className="hidden"
          />
        </label>
      </div>

      {/* TOC list */}
      <div className="flex-1 overflow-y-auto p-2">
        {state.toc.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 pb-8">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center text-[1.5rem]
              bg-veda-accent/8 dark:bg-veda-accent-dark/10">
              📄
            </div>
            <p className="veda-text-muted text-xs text-center leading-relaxed">
              Upload a PDF to see<br />the table of contents
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-0.5">
            {state.toc.map((item, i) => {
              const isActive = activeItem === item
              return (
                <button
                  key={i}
                  onClick={() => dispatch({ type: "SET_PAGE", payload: item.page })}
                  className={cn(
                    'flex items-start gap-3 w-full py-2.5 px-3 rounded-md border-none text-left mb-0.5 transition-all duration-200 cursor-pointer',
                    isActive
                      ? 'bg-veda-accent-soft border-l-2 border-veda-accent dark:bg-[#2a2a2a] dark:border-l-veda-accent-dark'
                      : 'bg-transparent border-l-2 border-transparent hover:bg-black/[0.03] dark:hover:bg-white/[0.03]'
                  )}
                >
                  <span className="veda-accent-text text-[0.7rem] font-semibold min-w-6 pt-0.5 shrink-0">
                    {item.printedPage ?? item.page}
                  </span>
                  <span
                    className={cn(
                      'text-[0.8rem] leading-relaxed break-words',
                      isActive
                        ? 'veda-text-primary font-semibold'
                        : 'text-[#6d6d6d] dark:text-[#aaaaaa] font-normal'
                    )}
                  >
                    {item.title}
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </div>
    </aside>
  )
}
