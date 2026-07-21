import { isSarvamCreditsPayload } from '../lib/sarvamErrors'
import API, { apiHeaders } from '../config/api'

/** Session cache for /page_layout — dedupes parallel and repeat fetches per page. */
const pageLayoutCache = new Map()
const pageLayoutInflight = new Map()

export function clearPageLayoutCache() {
  pageLayoutCache.clear()
  pageLayoutInflight.clear()
}

export async function fetchPageLayout(pageNumber) {
  if (!pageNumber || pageNumber < 1) {
    return null
  }
  if (pageLayoutCache.has(pageNumber)) {
    return pageLayoutCache.get(pageNumber)
  }
  if (pageLayoutInflight.has(pageNumber)) {
    return pageLayoutInflight.get(pageNumber)
  }

  const promise = (async () => {
    try {
      const response = await fetch(API.pageLayout(pageNumber), { headers: apiHeaders() })
      if (!response.ok) {
        return null
      }
      const data = await response.json()
      pageLayoutCache.set(pageNumber, data)
      return data
    } catch (err) {
      console.warn('Backend layout fetch failed:', err)
      return null
    } finally {
      pageLayoutInflight.delete(pageNumber)
    }
  })()

  pageLayoutInflight.set(pageNumber, promise)
  return promise
}

function isPageNumber(str) {
  const trimmed = str.trim()
  return /^\d+$/.test(trimmed) || /^[ivx]+$/i.test(trimmed)
}

function cleanSplitWords(text) {
  if (!text) return ''
  return text
    // 1. Fix hyphens split across lines or spaces (e.g. "expect- ed" -> "expected", "strik- ing" -> "striking")
    .replace(/([a-zA-Z]+)-\s+([a-zA-Z]+)/g, '$1$2')
    .replace(/([a-zA-Z]+)-\s+(\d+)/g, '$1-$2') // e.g. "Agni- 6" -> "Agni-6"
    // 2. Fix common split words due to font/kerning gaps
    .replace(/\bmys\s+tery\b/gi, 'mystery')
    .replace(/\bchromos\s+omes\b/gi, 'chromosomes')
    .replace(/\bdeve\s+lopment\b/gi, 'development')
    .replace(/\bdeve\s+lop\b/gi, 'develop')
    .replace(/\benve\s+lope\b/gi, 'envelope')
    .replace(/\bcapa\s+bility\b/gi, 'capability')
    .replace(/\bcapa\s+bilities\b/gi, 'capabilities')
    .replace(/\bintercon\s+tinental\b/gi, 'intercontinental')
    .replace(/\bcharac\s+ter\b/gi, 'character')
    .replace(/\bcharac\s+teristic\b/gi, 'characteristic')
    .replace(/\bcharac\s+teristics\b/gi, 'characteristics')
}

function normalizeText(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function splitIntoSentences(text) {
  return text
    .replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
}

function buildLines(items) {
  const lines = []
  let lastY = null

  for (const item of items) {
    if (lastY !== null && Math.abs(item.transform[5] - lastY) > 2) {
      lines.push('')
    }

    if (lines.length === 0) {
      lines.push('')
    }

    lines[lines.length - 1] += item.str
    lastY = item.transform[5]
  }

  return lines
}

function buildStructuredLines(items) {
  const lines = []
  let currentLine = null

  for (const item of items) {
    const text = item.str?.trim()

    if (!text) {
      continue
    }

    const y = Math.round(item.transform[5] * 10) / 10
    const x = item.transform[4]
    const fontSize = Math.max(item.height || 0, Math.abs(item.transform[0]) || 0)

    if (!currentLine || Math.abs(currentLine.y - y) > 3) {
      currentLine = {
        y,
        minX: x,
        maxFontSize: fontSize,
        segments: [{ x, text }],
      }
      lines.push(currentLine)
      continue
    }

    currentLine.minX = Math.min(currentLine.minX, x)
    currentLine.maxFontSize = Math.max(currentLine.maxFontSize, fontSize)
    currentLine.segments.push({ x, text })
  }

  return lines
    .map((line) => {
      const orderedSegments = line.segments.sort((a, b) => a.x - b.x)
      let text = ''

      for (const segment of orderedSegments) {
        if (!text) {
          text = segment.text
          continue
        }

        const needsSpace = !text.endsWith('-') && !segment.text.startsWith(',') && !segment.text.startsWith('.')
        text += needsSpace ? ` ${segment.text}` : segment.text
      }

      return {
        text: cleanSplitWords(text.replace(/\s+/g, ' ').trim()),
        y: line.y,
        x: line.minX,
        fontSize: line.maxFontSize,
      }
    })
    .filter((line) => line.text.length > 0)
}

function dedupeLines(lines) {
  const seen = new Set()

  return lines.filter((line) => {
    const normalized = normalizeText(line.text)

    if (!normalized || seen.has(normalized)) {
      return false
    }

    seen.add(normalized)
    return true
  })
}

function cleanHeadlineText(text) {
  if (!text) return ''
  return text
    .replace(/^\.\.\.\s*/, '')
    .replace(/\s*\.\.\.\s*$/, '')
    .replace(/\s*\.\.\.\s+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Collapse lines where one normalized text is a substring of another. */
function collapsePartialDuplicateLines(lines) {
  const result = []

  for (const line of lines) {
    const norm = normalizeText(line.text)
    if (!norm) continue

    let replaced = false
    for (let i = 0; i < result.length; i++) {
      const existingNorm = normalizeText(result[i].text)
      if (existingNorm === norm) {
        replaced = true
        break
      }
      if (existingNorm.includes(norm) || norm.includes(existingNorm)) {
        if (norm.length >= existingNorm.length) {
          result[i] = line
        }
        replaced = true
        break
      }
    }

    if (!replaced) {
      result.push(line)
    }
  }

  return result
}

/**
 * Merge vertically stacked magazine headline lines (e.g. "IS IT REALLY TRUE THAT …"
 * + "… stars are twinkling?") into a single line.
 */
function mergeStackedHeadlineLines(lines) {
  if (lines.length <= 1) return lines

  const sorted = [...lines].sort((a, b) => b.y - a.y)
  const maxFont = Math.max(...sorted.map((l) => l.fontSize))
  const headlineMinSize = maxFont * 0.82
  const consumed = new Set()
  const mergedLines = []

  for (let i = 0; i < sorted.length; i++) {
    if (consumed.has(i)) continue

    const line = sorted[i]
    const wordCount = line.text.split(/\s+/).filter(Boolean).length
    const isHeadline = line.fontSize >= headlineMinSize && wordCount <= 14

    if (!isHeadline) {
      mergedLines.push(line)
      continue
    }

    const parts = [line]
    consumed.add(i)

    for (let j = i + 1; j < sorted.length; j++) {
      if (consumed.has(j)) continue

      const next = sorted[j]
      const nextWords = next.text.split(/\s+/).filter(Boolean).length
      const vertGap = Math.abs(parts[parts.length - 1].y - next.y)
      const similarSize = Math.abs(parts[0].fontSize - next.fontSize) <= 2.5
      const continuation =
        /^\.\.\./.test(next.text.trim()) ||
        /\.\.\.\s*$/.test(parts[parts.length - 1].text.trim())
      const isNextHeadline = next.fontSize >= headlineMinSize && nextWords <= 14

      if (vertGap <= 90 && similarSize && isNextHeadline && (continuation || vertGap <= 70)) {
        parts.push(next)
        consumed.add(j)
      }
    }

    if (parts.length > 1) {
      mergedLines.push({
        text: cleanHeadlineText(parts.map((p) => p.text).join(' ')),
        x: Math.min(...parts.map((p) => p.x)),
        y: Math.max(...parts.map((p) => p.y)),
        fontSize: Math.max(...parts.map((p) => p.fontSize)),
        _mergedHeadline: true,
      })
    } else {
      mergedLines.push(line)
    }
  }

  return sortLinesByReadingOrder(mergedLines)
}

/** Magazine Q&A / "Did you know?" callout pages (large question + multi-column body). */
function detectCalloutPage(lines) {
  if (lines.length < 5) return false

  const maxFont = Math.max(...lines.map((l) => l.fontSize))
  const topLines = [...lines].sort((a, b) => b.y - a.y).slice(0, 5)
  const hasLargeHeadline = topLines.some((l) => l.fontSize >= maxFont * 0.85)
  const hasQuestion = lines.some(
    (l) =>
      /\?\s*$/.test(l.text.trim()) ||
      /is it really true/i.test(l.text) ||
      l._mergedHeadline
  )
  const xSpread = Math.max(...lines.map((l) => l.x)) - Math.min(...lines.map((l) => l.x))

  return hasLargeHeadline && hasQuestion && xSpread > 100
}

function lineRoleForAnalyze(line, { headlineMin, subheadMin }) {
  const words = line.text.split(/\s+/).filter(Boolean).length
  if (line._mergedHeadline || (line.fontSize >= headlineMin && words <= 16)) {
    return 'HEADLINE'
  }
  if (line.fontSize >= subheadMin && words <= 18) {
    return 'SUBHEAD'
  }
  if (words <= 8 && line.fontSize >= subheadMin * 0.92) {
    return 'CALLOUT'
  }
  return 'BODY'
}

/** Format extracted lines with layout roles for the LLM (preserves structure). */
function buildStructuredAnalyzeText(lines, { isCallout = false } = {}) {
  if (!lines.length) return ''

  const sizes = lines.map((l) => l.fontSize)
  const maxFont = Math.max(...sizes)
  const median = [...sizes].sort((a, b) => a - b)[Math.floor(sizes.length / 2)] || 12
  const headlineMin = maxFont * 0.82
  const subheadMin = Math.max(median * 1.2, maxFont * 0.55)

  const ordered = sortLinesByReadingOrder(lines)
  const roles = ordered.map((line) =>
    lineRoleForAnalyze(line, { headlineMin, subheadMin })
  )

  const header = isCallout
    ? 'Page layout: magazine Q&A callout. Merge all HEADLINE lines into one natural question title.\n\n'
    : 'Page layout: structured extract. Lines are tagged by role — use HEADLINE for title, SUBHEAD for subtitle.\n\n'

  const body = ordered
    .map((line, idx) => `[${roles[idx]}] ${line.text.trim()}`)
    .join('\n')

  return `${header}${body}`.trim()
}

function buildTitle(lines, fallbackTitle) {
  const mergedHeadline = lines.find((line) => line._mergedHeadline)
  if (mergedHeadline?.text) {
    return cleanHeadlineText(mergedHeadline.text)
  }

  const candidates = [...lines]
    .filter((line) => line.text.length > 4 && !/discovery|november|contents/i.test(line.text))
    .sort((a, b) => b.fontSize - a.fontSize || a.y - b.y)

  if (candidates.length > 0) {
    return candidates[0].text
  }

  return fallbackTitle || 'Untitled section'
}

function buildSubtitle(lines, title) {
  const normalizedTitle = normalizeText(title)

  const candidates = lines
    .filter((line) => {
      const norm = normalizeText(line.text)
      if (!norm || norm === normalizedTitle) return false
      if (line._mergedHeadline) return false
      if (normalizedTitle.includes(norm) && norm.length > 12) return false
      if (norm.includes(normalizedTitle) && normalizedTitle.length > 12) return false
      return line.text.length > 10
    })
    .sort((a, b) => b.fontSize - a.fontSize || a.y - b.y)

  return candidates[0]?.text ?? ''
}

function buildHighlights(sentences, summarySentences = []) {
  const summarySet = new Set(summarySentences.map((s) => s.toLowerCase().trim()))
  return sentences
    .filter((sentence) => sentence.length > 35 && !summarySet.has(sentence.toLowerCase().trim()))
    .slice(0, 3)
}

function buildNarration(title, summary) {
  const intro = `This section is about ${title}.`
  const summaryPart = summary ? ` ${summary}` : ''
  return `${intro}${summaryPart}`.replace(/\s+/g, ' ').trim()
}


function sortLinesByReadingOrder(lines) {
  if (lines.length <= 1) return lines

  // Calculate page bounds and adaptive column threshold (20% of active width, at least 60px)
  const xValues = lines.map((l) => l.x)
  const minX = Math.min(...xValues)
  const maxX = Math.max(...xValues)
  const activeWidth = maxX - minX
  const colThreshold = Math.max(60, activeWidth * 0.20)

  // 1. Sort all lines by x ascending
  const sortedByX = [...lines].sort((a, b) => a.x - b.x)

  // 2. Group into columns based on adaptive x coordinate differences from the first element of the column
  const columns = []
  let currentColumn = [sortedByX[0]]
  columns.push(currentColumn)

  for (let i = 1; i < sortedByX.length; i++) {
    const line = sortedByX[i]
    const firstLine = currentColumn[0]
    if (line.x - firstLine.x < colThreshold) {
      currentColumn.push(line)
    } else {
      currentColumn = [line]
      columns.push(currentColumn)
    }
  }

  // 3. Within each column, sort by y descending (top-to-bottom)
  for (const col of columns) {
    col.sort((a, b) => b.y - a.y)
  }

  // 4. Flatten the columns back to a single array
  return columns.flat()
}

/**
 * Detect distinct topic blocks on a page by finding lines whose font size is
 * significantly larger than the median body text (i.e. they are headings).
 * Returns an array of { title, body, sentences } objects — one per topic.
 */
function detectTopics(lines, fallbackTitle = '') {
  if (!lines.length) return []

  // Pre-sort the input lines by column-based reading order!
  const sortedLines = sortLinesByReadingOrder(lines)

  // Compute heading threshold using font-size gap clustering.
  // Strategy: find the largest absolute jump between consecutive distinct font
  // sizes. The threshold sits at the midpoint of that gap — everything above is
  // treated as a heading. Fallback to the old median-based formula when the gap
  // is too small (< 2pt) to be meaningful (e.g. all body text, no real headings).
  const sizes = [...sortedLines].map((l) => l.fontSize).sort((a, b) => a - b)
  const median = sizes[Math.floor(sizes.length / 2)] || 1
  const maxFontSize = Math.max(...sizes)

  const uniqueSizes = [...new Set(sizes)].sort((a, b) => a - b)
  let maxGap = 0
  let gapMidpoint = null
  for (let i = 1; i < uniqueSizes.length; i++) {
    const gap = uniqueSizes[i] - uniqueSizes[i - 1]
    if (gap > maxGap) {
      maxGap = gap
      gapMidpoint = (uniqueSizes[i - 1] + uniqueSizes[i]) / 2
    }
  }
  // Only use gap-based threshold if the gap is at least 2pt and splits
  // meaningfully (threshold must be above the body-text median).
  const headingThreshold =
    gapMidpoint && maxGap >= 2 && gapMidpoint > median
      ? gapMidpoint
      : Math.max(median * 1.35, maxFontSize * 0.50)

  const normalizedFallback = fallbackTitle ? normalizeText(fallbackTitle) : ''

  // A heading line: larger font AND short enough to be a title (≤ 14 words)
  const isHeading = (line) =>
    line.fontSize >= headingThreshold &&
    line.text.split(/\s+/).length <= 14 &&
    (!normalizedFallback || normalizeText(line.text) !== normalizedFallback)

  const topics = []
  let currentTopic = null
  // Body lines that appear before any heading in the PDF content stream.
  const orphanLines = []

  for (const line of sortedLines) {
    if (isHeading(line)) {
      // ── Fix A: merge consecutive heading lines ─────────────────────────
      const isCloseVertically = currentTopic && Math.abs(currentTopic.lastY - line.y) < 75 && Math.abs(currentTopic.x - line.x) < 150;
      const isSimilarSize = currentTopic && Math.abs(currentTopic.fontSize - line.fontSize) <= 1.5;

      if (currentTopic && currentTopic.bodyLines.length === 0 && (isSimilarSize || isCloseVertically)) {
        // Prevent substring duplicates during consecutive title merges
        if (!normalizeText(currentTopic.title).includes(normalizeText(line.text))) {
          currentTopic.title = `${currentTopic.title} ${line.text}`.replace(/\s+/g, ' ').trim()
        }
        currentTopic.lastY = line.y
      } else {
        currentTopic = { title: line.text, fontSize: line.fontSize, x: line.x, y: line.y, lastY: line.y, bodyLines: [] }
        topics.push(currentTopic)
      }
    } else {
      if (currentTopic) {
        currentTopic.bodyLines.push(line.text)
      } else {
        // ── Fix B: collect orphan body lines ─────────────────────────────
        orphanLines.push(line.text)
      }
    }
  }

  // ── Fix B cont.: do not assign orphan lines to the last topic; return them as intro instead
  // if (orphanLines.length > 0 && topics.length > 0) {
  //   topics[topics.length - 1].bodyLines.push(...orphanLines)
  // }

  // Deduplicate topics (e.g. if one title is a substring of another)
  const dedupedTopics = []
  for (const topic of topics) {
    const norm = normalizeText(topic.title)
    let duplicateIdx = -1
    for (let i = 0; i < dedupedTopics.length; i++) {
      const existingNorm = normalizeText(dedupedTopics[i].title)
      if (norm.includes(existingNorm) || existingNorm.includes(norm)) {
        duplicateIdx = i
        break
      }
    }

    if (duplicateIdx !== -1) {
      const existing = dedupedTopics[duplicateIdx]
      // Merge body lines
      const mergedBodyLines = [...existing.bodyLines, ...topic.bodyLines]
      // Keep the one with the larger font size
      if (topic.fontSize > existing.fontSize) {
        existing.title = topic.title
        existing.fontSize = topic.fontSize
        existing.x = topic.x
        existing.y = topic.y
        existing.lastY = topic.lastY
      }
      existing.bodyLines = mergedBodyLines
    } else {
      dedupedTopics.push(topic)
    }
  }

  // Convert bodyLines → body string + sentences
  const detected = dedupedTopics
    .map((t) => {
      const body = t.bodyLines.join(' ').replace(/\s+/g, ' ').trim()
      const sentences = splitIntoSentences(body)
      return { title: t.title, fontSize: t.fontSize, x: t.x, y: t.y, body, sentences }
    })
    .filter((t) => t.body.length > 0)

  const introText = orphanLines.join(' ').replace(/\s+/g, ' ').trim()

  if (detected.length <= 1) {
    return { topics: detected, intro: introText }
  }

  // Sort by layout order: vertical rows top-to-bottom, columns left-to-right
  return { topics: sortTopicsByLayout(detected), intro: introText }
}

/**
 * Build a narration script that briefly covers every topic in a digest page.
 */
function buildDigestNarration(topics) {
  const intro = `This page covers ${topics.length} topic${topics.length !== 1 ? 's' : ''}.`
  const parts = topics.map((t) => {
    const firstSentence = t.sentences[0] || t.body.slice(0, 120)
    return `${t.title}: ${firstSentence}`
  })
  return `${intro} ${parts.join(' ')}`.replace(/\s+/g, ' ').trim()
}

function parseTOCEntriesFromLines(lines) {
  const pattern = /^(\d+)\.\s+(.+)$/
  const rawEntries = []

  for (const line of lines) {
    const trimmed = line.trim()
    const match = trimmed.match(pattern)

    if (match) {
      rawEntries.push({
        printedPage: parseInt(match[1], 10),
        title: cleanSplitWords(match[2].trim()),
      })
    } else if (rawEntries.length > 0 && trimmed.length > 0 && !/^\d/.test(trimmed)) {
      const last = rawEntries[rawEntries.length - 1]

      if (last.title.length < 80) {
        const needsSpace = !last.title.endsWith('-')
        const rawTitle = needsSpace
          ? `${last.title} ${trimmed}`.replace(/\s+/g, ' ').trim()
          : `${last.title.slice(0, -1)}${trimmed}`.replace(/\s+/g, ' ').trim()
        last.title = cleanSplitWords(rawTitle)
      }
    }
  }

  return rawEntries
}

function shouldContinueTOC(previousEntries, nextEntries, pageText) {
  if (!nextEntries.length) {
    return false
  }

  if (/contents/i.test(pageText)) {
    return true
  }

  if (!previousEntries.length) {
    return nextEntries.length > 3
  }

  const previousMax = Math.max(...previousEntries.map((entry) => entry.printedPage))
  const nextMin = Math.min(...nextEntries.map((entry) => entry.printedPage))

  return nextMin > previousMax
}

function mergeTOCEntries(entries) {
  const seen = new Set()

  return entries.filter((entry) => {
    const key = `${entry.printedPage}:${entry.title}`

    if (seen.has(key)) {
      return false
    }

    seen.add(key)
    return true
  })
}

async function getPageTextFromPdfJs(pdfDoc, pageNumber) {
  if (!pageNumber || pageNumber < 1 || pageNumber > pdfDoc.numPages) {
    return ''
  }

  const page = await pdfDoc.getPage(pageNumber)
  const textContent = await page.getTextContent()
  const viewport = page.getViewport({ scale: 1.0 })
  const pageHeight = viewport.height

  const rawText = textContent.items
    .filter((item) => {
      if (isPageNumber(item.str)) {
        return false
      }
      const y = item.transform[5]
      if (y > pageHeight * 0.91) {
        return false
      }
      if (y < pageHeight * 0.06) {
        return false
      }
      return true
    })
    .map((item) => item.str)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim()

  return cleanSplitWords(rawText)
}

async function getPageText(pdfDoc, pageNumber) {
  if (!pageNumber || pageNumber < 1 || pageNumber > pdfDoc.numPages) {
    return ''
  }

  const data = await fetchPageLayout(pageNumber)
  if (data?.lines) {
    const rawText = data.lines.map((line) => line.text).join(' ').replace(/\s+/g, ' ').trim()
    return cleanSplitWords(rawText)
  }

  return getPageTextFromPdfJs(pdfDoc, pageNumber)
}

function buildFallbackTOC(pdfDoc) {
  return Array.from({ length: pdfDoc.numPages }, (_, i) => ({
    printedPage: i + 1,
    page: i + 1,
    title: `Page ${i + 1}`,
  }))
}

async function detectPageOffset(pdfDoc, tocEntries) {
  if (!tocEntries.length) {
    return 0
  }

  const maxPrintedPage = Math.max(...tocEntries.map((entry) => entry.printedPage))
  const maxOffset = Math.max(0, Math.min(12, pdfDoc.numPages - maxPrintedPage))
  const sampleEntries = tocEntries.slice(0, Math.min(6, tocEntries.length))
  let bestOffset = 0
  let bestScore = -1

  for (let offset = 0; offset <= maxOffset; offset += 1) {
    let score = 0

    for (const entry of sampleEntries) {
      const actualPage = entry.printedPage + offset
      const pageText = normalizeText(await getPageTextFromPdfJs(pdfDoc, actualPage))
      const title = normalizeText(entry.title)

      if (!pageText || !title) {
        continue
      }

      if (pageText.includes(title)) {
        score += 3
        continue
      }

      const titleWords = title.split(' ').filter(Boolean)
      const wordMatches = titleWords.filter((word) => word.length > 3 && pageText.includes(word)).length

      if (titleWords.length > 0) {
        score += wordMatches / titleWords.length
      }
    }

    if (score > bestScore) {
      bestScore = score
      bestOffset = offset
    }
  }

  return bestOffset
}

export async function extractTOC(pdfDoc) {
  try {
    let collectedEntries = []
    let tocStarted = false

    for (let p = 1; p <= Math.min(10, pdfDoc.numPages); p += 1) {
      const page = await pdfDoc.getPage(p)
      const content = await page.getTextContent()
      const lines = buildLines(content.items)
      const pageText = lines.join(' ')
      const tocEntries = parseTOCEntriesFromLines(lines)
        .filter((entry) => entry.printedPage > 0 && entry.title.length > 1)
        .sort((a, b) => a.printedPage - b.printedPage)

      if (!tocStarted && tocEntries.length > 3) {
        tocStarted = true
        collectedEntries = tocEntries
        continue
      }

      if (tocStarted && shouldContinueTOC(collectedEntries, tocEntries, pageText)) {
        collectedEntries = mergeTOCEntries([...collectedEntries, ...tocEntries])
        continue
      }

      if (tocStarted) {
        break
      }
    }

    if (collectedEntries.length > 3) {
      const offset = await detectPageOffset(pdfDoc, collectedEntries)

      return {
        offset,
        toc: collectedEntries
          .map((entry) => ({
            ...entry,
            page: entry.printedPage + offset,
          }))
          .filter((entry) => entry.page > 0 && entry.page <= pdfDoc.numPages),
      }
    }
  } catch (error) {
    console.error(error)
  }

  return {
    offset: 0,
    toc: buildFallbackTOC(pdfDoc),
  }
}

function getPdfObject(page, id) {
  return new Promise((resolve) => {
    let resolved = false
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true
        resolve(null)
      }
    }, 150)

    try {
      page.objs.get(id, (obj) => {
        if (!resolved) {
          resolved = true
          clearTimeout(timeout)
          resolve(obj)
        }
      })
    } catch {
      if (!resolved) {
        resolved = true
        clearTimeout(timeout)
        resolve(null)
      }
    }
  })
}

function getPdfCommonObject(page, id) {
  return new Promise((resolve) => {
    let resolved = false
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true
        resolve(null)
      }
    }, 150)

    try {
      page.commonObjs.get(id, (obj) => {
        if (!resolved) {
          resolved = true
          clearTimeout(timeout)
          resolve(obj)
        }
      })
    } catch {
      if (!resolved) {
        resolved = true
        clearTimeout(timeout)
        resolve(null)
      }
    }
  })
}

async function extractPageImages(page) {
  const images = []
  try {
    const opList = await page.getOperatorList()
    const PAINT_IMAGE = 85
    const PAINT_INLINE_IMAGE = 86

    let currentTransform = [1, 0, 0, 1, 0, 0]

    for (let i = 0; i < opList.fnArray.length && images.length < 20; i++) {
      const fn = opList.fnArray[i]
      if (fn === 12) { // transform
        currentTransform = opList.argsArray[i]
      }

      if (fn === PAINT_IMAGE || fn === PAINT_INLINE_IMAGE) {
        let img = null
        if (fn === PAINT_IMAGE) {
          const imageName = opList.argsArray[i][0]
          img = await getPdfObject(page, imageName)
          if (!img) {
            img = await getPdfCommonObject(page, imageName)
          }
        } else {
          // Inline image
          img = opList.argsArray[i][0]
        }

        if (img && img.width > 120 && img.height > 120) {
          const canvas = document.createElement('canvas')
          canvas.width = img.width
          canvas.height = img.height
          const ctx = canvas.getContext('2d')
          
          const isDrawable = (typeof ImageBitmap !== 'undefined' && img instanceof ImageBitmap) || 
                             (typeof HTMLImageElement !== 'undefined' && img instanceof HTMLImageElement) || 
                             (typeof HTMLCanvasElement !== 'undefined' && img instanceof HTMLCanvasElement)

          let dataUrl = null
          if (img.bitmap) {
            ctx.drawImage(img.bitmap, 0, 0)
            dataUrl = canvas.toDataURL('image/png')
          } else if (isDrawable) {
            ctx.drawImage(img, 0, 0)
            dataUrl = canvas.toDataURL('image/png')
          } else if (img.data instanceof Uint8ClampedArray || img.data instanceof Uint8Array) {
            const imageData = ctx.createImageData(img.width, img.height)
            const length = img.data.length
            const pixelCount = img.width * img.height
            
            if (length === pixelCount * 4) {
              imageData.data.set(img.data)
            } else if (length === pixelCount * 3) {
              let srcIdx = 0
              let destIdx = 0
              for (let p = 0; p < pixelCount; p++) {
                imageData.data[destIdx] = img.data[srcIdx]
                imageData.data[destIdx + 1] = img.data[srcIdx + 1]
                imageData.data[destIdx + 2] = img.data[srcIdx + 2]
                imageData.data[destIdx + 3] = 255
                srcIdx += 3
                destIdx += 4
              }
            } else if (length === pixelCount) {
              let destIdx = 0
              for (let p = 0; p < pixelCount; p++) {
                const val = img.data[p]
                imageData.data[destIdx] = val
                imageData.data[destIdx + 1] = val
                imageData.data[destIdx + 2] = val
                imageData.data[destIdx + 3] = 255
                destIdx += 4
              }
            } else {
              continue
            }
            
            ctx.putImageData(imageData, 0, 0)
            dataUrl = canvas.toDataURL('image/png')
          }

          if (dataUrl) {
            images.push({
              url: dataUrl,
              x: currentTransform[4],
              y: currentTransform[5]
            })
          }
        }
      }
    }
  } catch (err) {
    console.error('Error in extractPageImages:', err)
  }
  return images
}

export async function extractPageText(pdfDoc, pageNumber) {
  return getPageText(pdfDoc, pageNumber)
}

export async function fetchSemanticAnalysis(sourceText, fallbackTitle = '', isDigest = false, language = 'en-US', options = {}) {
  const { onSarvamCreditsExhausted, structuredText, pageLayout = 'plain' } = options
  const analyzeText = (structuredText || sourceText || '').trim()
  // If the page contains very little content (under 50 words), skip the local LLM
  // to avoid hallucinations or system instructions reflections.
  const wordCount = analyzeText ? analyzeText.trim().split(/\s+/).filter(Boolean).length : 0
  if (wordCount < 50) {
    return null
  }

  try {
    const ANALYZE_URL = API.analyze
    const response = await fetch(ANALYZE_URL, {
      method: 'POST',
      headers: apiHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        text: analyzeText,
        is_digest: isDigest,
        language,
        page_layout: pageLayout,
      }),
    })

    if (response.ok) {
      const data = await response.json()

      if (data.sarvam_credits_exhausted && onSarvamCreditsExhausted) {
        onSarvamCreditsExhausted(data)
      }

      // ── Digest response from LLM ───────────────────────────────────────────
      if (isDigest && Array.isArray(data.topics)) {
        return {
          isDigest: true,
          title: data.title || fallbackTitle,
          subtitle: data.subtitle || `${data.topics.length} stories`,
          summary: data.summary || `A roundup of ${data.topics.length} stories on this page.`,
          topics: data.topics.map((t) => ({ 
            ...t, 
            body: t.body || t.summary || '',
            narration_te: t.narration_te || null 
          })),
          highlights: data.topics.map((t) => `${t.title}: ${t.summary}`),
          supportingPoints: data.topics.flatMap((t) => (t.summary ? [t.summary] : [])),
          narration: data.narration || buildDigestNarration(
            data.topics.map((t) => ({ title: t.title, sentences: [t.summary || ''] }))
          ),
          narration_te: data.narration_te || null,
        }
      }
      // ── Single-topic response ──────────────────────────────────────────────
      return {
        isDigest: false,
        title: data.title || fallbackTitle,
        subtitle: data.subtitle || '',
        summary: data.summary || '',
        highlights: data.highlights || [],
        supportingPoints: data.supportingPoints || [],
        narration: data.narration || buildNarration(data.title || fallbackTitle, data.summary || '', data.highlights || []),
        narration_te: data.narration_te || null,
      }
    } else {
      // FastAPI raises HTTPException with a "detail" field (not "error")
      const errData = await response.json().catch(() => ({}))
      if (onSarvamCreditsExhausted && isSarvamCreditsPayload(errData, response.status)) {
        onSarvamCreditsExhausted(errData)
      }
      const msg = errData.detail || errData.error || `HTTP ${response.status}`
      console.warn('Local LLM analysis returned error:', msg)
    }
  } catch (err) {
    // Network errors (server offline, timeout, etc.) are expected when the local LLM
    // model hasn't been downloaded yet — suppress the noisy stack trace
    if (err instanceof TypeError && err.message.includes('fetch')) {
      console.warn('Local LLM analysis unavailable (server offline or model not loaded)')
    } else {
      console.warn('Local LLM analysis failed:', err)
    }
  }
  return null
}

export async function fetchTranslatedFields(deck, fields = ['highlights', 'supportingPoints'], language = 'te-IN') {
  if (!deck) return null
  try {
    const apiBase = API.base
    const response = await fetch(API.translateFields, {
      method: 'POST',
      headers: apiHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ deck, fields, language }),
    })
    if (!response.ok) return null
    return await response.json()
  } catch {
    return null
  }
}

export async function fetchTeluguDeck(deck, options = {}) {
  const { onSarvamCreditsExhausted } = options
  if (!deck) return null

  try {
    const apiBase = API.base
    const response = await fetch(API.translateDeck, {
      method: 'POST',
      headers: apiHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ deck }),
    })

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}))
      if (onSarvamCreditsExhausted && isSarvamCreditsPayload(errData, response.status)) {
        onSarvamCreditsExhausted(errData)
      }
      console.warn('Telugu deck translation failed:', errData.detail || response.status)
      return null
    }

    const translated = await response.json()
    const mergedTopics = Array.isArray(translated.topics)
      ? translated.topics.map((topic, idx) => ({
        ...topic,
        body: topic.body || topic.summary || '',
        image: deck.topics?.[idx]?.image || topic.image || null,
      }))
      : deck.topics

    return {
      ...deck,
      ...translated,
      topics: mergedTopics || deck.topics,
      images: deck.images || translated.images,
      isTelugu: true,
    }
  } catch (err) {
    console.warn('Telugu deck translation unavailable:', err)
    return null
  }
}

function prepareLayoutLines(rawLines) {
  let lines = dedupeLines(rawLines || [])
  lines = collapsePartialDuplicateLines(lines)
  lines = mergeStackedHeadlineLines(lines)
  return lines
}

/** Inset figures only — excludes full-page PDF backgrounds. */
function foregroundLayoutImages(imagesData) {
  return (imagesData || []).filter((img) => !img.isBackground)
}

/** Foreground images for carousels; fall back to largest background on cover pages. */
function pickDisplayImages(imagesData) {
  const foreground = foregroundLayoutImages(imagesData)
  if (foreground.length) return foreground
  const backgrounds = (imagesData || []).filter((img) => img.isBackground)
  if (!backgrounds.length) return []
  return [...backgrounds].sort(
    (a, b) => (b.w || 0) * (b.h || 0) - (a.w || 0) * (a.h || 0)
  )
}

export async function extractPagePresentation(pdfDoc, pageNumber, fallbackTitle = '') {
  if (!pageNumber || pageNumber < 1 || pageNumber > pdfDoc.numPages) {
    return null
  }

  let lines = []
  let imagesData = []
  let images = []
  let sourceText = ''
  let analyzeText = ''
  let pageLayout = 'plain'
  let pageWidth = 0
  let pageHeight = 0
  let fetchedBackend = false

  const layoutData = await fetchPageLayout(pageNumber)
  if (layoutData?.lines) {
    lines = prepareLayoutLines(layoutData.lines)
    const rawSourceText = lines.map((line) => line.text).join(' ').replace(/\s+/g, ' ').trim()
    sourceText = cleanSplitWords(rawSourceText)
    analyzeText = buildStructuredAnalyzeText(lines, {
      isCallout: detectCalloutPage(lines),
    })
    pageLayout = detectCalloutPage(lines) ? 'qa' : 'structured'
    pageWidth = layoutData.width || 0
    pageHeight = layoutData.height || 0
    imagesData = layoutData.images || []
    images = pickDisplayImages(imagesData).map((img) => img.url)
    fetchedBackend = true
  }

  // 2. Fallback to frontend-only if backend failed
  if (!fetchedBackend) {
    const page = await pdfDoc.getPage(pageNumber)
    const textContent = await page.getTextContent()
    const viewport = page.getViewport({ scale: 1.0 })
    const pageHeight = viewport.height

    const filteredItems = textContent.items.filter((item) => {
      if (isPageNumber(item.str)) {
        return false
      }
      const y = item.transform[5]
      if (y > pageHeight * 0.91) {
        return false // Exclude top headers
      }
      if (y < pageHeight * 0.06) {
        return false // Exclude footers
      }
      return true // Include body text
    })

    lines = prepareLayoutLines(buildStructuredLines(filteredItems))
    const rawSourceText = lines.map((line) => line.text).join(' ').replace(/\s+/g, ' ').trim()
    sourceText = cleanSplitWords(rawSourceText)
    analyzeText = buildStructuredAnalyzeText(lines, {
      isCallout: detectCalloutPage(lines),
    })
    pageLayout = detectCalloutPage(lines) ? 'qa' : 'structured'
    imagesData = await extractPageImages(page)
    images = imagesData.map((img) => img.url)
  }

  // ── Image-primary page: has images but very little text ──────────────────
  // Cover pages, full-page ads, photo spreads. Skip LLM; build a minimal slide
  // that leads with the image.
  const wordCount = sourceText.trim().split(/\s+/).filter(Boolean).length
  if ((imagesData || []).length > 0 && wordCount < 50) {
    images = pickDisplayImages(imagesData).map((img) => img.url)
    const title = buildTitle(lines, fallbackTitle) || fallbackTitle || 'Visual'
    return {
      isDigest: false,
      isImagePrimary: true,
      title,
      subtitle: '',
      summary: sourceText || '',
      highlights: [],
      supportingPoints: [],
      sourceText,
      images,
      imagesMeta: imagesData,
      pageWidth,
      pageHeight,
      narration: title ? `This section features ${title}.` : 'Image-focused content.',
    }
  }

  // ── Multi-topic digest detection ──────────────────────────────────────────
  const { topics, intro } = detectTopics(lines, fallbackTitle)
  if (topics.length >= 3 || (topics.length === 2 && Math.abs(topics[0].fontSize - topics[1].fontSize) <= 2)) {
    // This is a digest page (e.g. a magazine "Science Updates" roundup)
    const sectionTitle = fallbackTitle || buildTitle(lines, 'News Digest')
    const highlights = topics.map((t) => {
      const s = t.sentences[0] || t.body.slice(0, 100)
      return `${t.title}: ${s}`
    })
    const supportingPoints = topics.flatMap((t) => t.sentences.slice(0, 2))
    const summary = intro || `A roundup of ${topics.length} stories on this page.`

    // Match each topic to inset figures only — skip full-page backgrounds
    const matchedImages = matchImagesToTopicsGlobal(topics, foregroundLayoutImages(imagesData))

    return {
      isDigest: true,
      title: sectionTitle,
      subtitle: `${topics.length} stories`,
      summary,
      topics: topics.map((t, idx) => ({
        title: t.title,
        body: t.sentences.slice(0, 4).join(' '),
        image: matchedImages[idx] || null
      })),
      highlights,
      supportingPoints,
      sourceText,
      analyzeText: buildStructuredAnalyzeText(lines),
      pageLayout: 'structured',
      images: matchedImages,
      imagesMeta: imagesData,
      pageWidth,
      pageHeight,
      narration: intro ? `${intro} ${buildDigestNarration(topics)}`.replace(/\s+/g, ' ').trim() : buildDigestNarration(topics),
    }
  }
  // ──────────────────────────────────────────────────────────────────────────

  // Single-topic heuristic fallback (unchanged)
  const title = buildTitle(lines, fallbackTitle)
  const subtitle = buildSubtitle(lines, title)
  const bodyLines = lines.filter(
    (line) =>
      !line._mergedHeadline &&
      normalizeText(line.text) !== normalizeText(title) &&
      normalizeText(line.text) !== normalizeText(subtitle)
  )
  const bodySourceText = bodyLines.map((l) => l.text).join(' ').replace(/\s+/g, ' ').trim()
  const sentences = splitIntoSentences(bodySourceText || sourceText)
  const summarySentences = sentences.slice(0, 2)
  const summary = summarySentences.join(' ')
  const highlights = buildHighlights(sentences, summarySentences)
  const supportingPoints = bodyLines
    .filter((line) => line.text.length > 18)
    .slice(0, 8)
    .map((line) => line.text)

  if (!analyzeText) {
    analyzeText = buildStructuredAnalyzeText(lines, { isCallout: detectCalloutPage(lines) })
    pageLayout = detectCalloutPage(lines) ? 'qa' : 'structured'
  }

  return {
    isDigest: false,
    title,
    subtitle,
    summary,
    highlights,
    supportingPoints,
    sourceText,
    analyzeText,
    pageLayout,
    images,
    imagesMeta: imagesData,
    pageWidth,
    pageHeight,
    narration: buildNarration(title, summary),
  }
}

// ── Helper: Segmented Block Sorter ──────────────────────────────────────────
function sortTopicsByLayout(topics) {
  if (topics.length <= 1) return topics

  // Estimate the vertical bounds of each topic.
  // In PDF coordinate space, y is bottom-up (meaning larger y values are at the top).
  // A topic starts at title.y (top) and its body lines flow downward.
  const topicsWithSpans = topics.map((t) => {
    // Average line height is roughly proportional to font size or standard spacing.
    const estimatedHeight = t.sentences ? t.sentences.length * 35 : 60
    return {
      ...t,
      top: t.y,
      bottom: t.y - estimatedHeight
    }
  })

  // Compare two topics to sort them.
  return [...topicsWithSpans].sort((a, b) => {
    // Check if they overlap vertically.
    // They do not overlap if one's bottom is above the other's top.
    const verticalOverlap = !(a.top < b.bottom || b.top < a.bottom)

    if (!verticalOverlap) {
      // If they don't overlap vertically (different rows), sort top-to-bottom (higher y comes first).
      return b.y - a.y
    } else {
      // If they do overlap vertically (columns), sort left-to-right (smaller x comes first).
      return a.x - b.x
    }
  })
}

// ── Helper: Vertical Zone Image Matcher ──────────────────────────────────────
// Each topic "owns" the vertical strip from its heading y down to the next
// heading y (or page bottom). An image belongs to the topic whose zone contains
// the image centre. This is more reliable than global distance minimisation for
// magazine-style multi-column digest layouts.
function matchImagesToTopicsGlobal(topics, imagesData) {
  const matchedImages = new Array(topics.length).fill(null)
  if (topics.length === 0 || imagesData.length === 0) return matchedImages

  // Sort topics descending by y (highest y = top of page in bottom-up coords).
  const sorted = topics
    .map((t, originalIdx) => ({ ...t, originalIdx }))
    .sort((a, b) => b.y - a.y)

  // Build vertical zones: each topic owns [zoneTop, zoneBottom].
  // zoneTop  = topic heading y (in bottom-up coords, larger = higher on page)
  // zoneBottom = next topic's heading y (or 0 = page bottom)
  const zones = sorted.map((t, i) => ({
    originalIdx: t.originalIdx,
    zoneTop: t.y,
    zoneBottom: i + 1 < sorted.length ? sorted[i + 1].y : 0,
    x: t.x,
  }))

  // Use image centre y (cy) if provided, fall back to image y.
  const usedImages = new Set()

  for (const zone of zones) {
    let bestImgIdx = -1
    let bestScore = -Infinity

    for (let i = 0; i < imagesData.length; i++) {
      if (usedImages.has(i)) continue
      const imgCy = imagesData[i].cy !== undefined ? imagesData[i].cy : imagesData[i].y
      const inZone = imgCy <= zone.zoneTop && imgCy >= zone.zoneBottom

      // Primary: image centre falls inside this topic's vertical zone.
      // Score = negative horizontal distance (prefer image in same column).
      if (inZone) {
        const score = -Math.abs(imagesData[i].x - zone.x)
        if (score > bestScore) { bestScore = score; bestImgIdx = i }
      }
    }

    if (bestImgIdx !== -1) {
      matchedImages[zone.originalIdx] = imagesData[bestImgIdx].url
      usedImages.add(bestImgIdx)
    }
  }

  // Second pass: assign leftover images to topics that are still null,
  // using nearest-centre distance so no image is completely orphaned.
  for (let i = 0; i < imagesData.length; i++) {
    if (usedImages.has(i)) continue
    const imgCy = imagesData[i].cy !== undefined ? imagesData[i].cy : imagesData[i].y
    let bestTopicIdx = -1
    let bestDist = Infinity
    for (let t = 0; t < topics.length; t++) {
      if (matchedImages[t] !== null) continue
      const d = Math.abs(topics[t].y - imgCy)
      if (d < bestDist) { bestDist = d; bestTopicIdx = t }
    }
    if (bestTopicIdx !== -1) {
      matchedImages[bestTopicIdx] = imagesData[i].url
      usedImages.add(i)
    }
  }

  return matchedImages
}

const API_BASE = API.base

/**
 * Upscale or regenerate images via the backend.
 * Decorative images may route to Kaggle SDXL; informational images use Lanczos.
 * Any failure falls back to the original URL.
 */
export async function upscaleImages(imageUrls, context = {}) {
  if (!imageUrls || imageUrls.length === 0) return imageUrls

  const {
    pageText = '',
    pageWidth = 0,
    pageHeight = 0,
    pageWordCount = 0,
    isImagePrimary = false,
    imagesMeta = [],
  } = context

  const metaByUrl = Object.fromEntries(
    (imagesMeta || []).filter((m) => m && m.url).map((m) => [m.url, m])
  )

  const results = await Promise.all(
    imageUrls.map(async (url) => {
      if (!url || !url.startsWith('data:image')) return url
      const meta = metaByUrl[url]
      if (meta?.isBackground) return url
      try {
        const fmt = url.startsWith('data:image/jpeg') ? 'jpeg' : 'png'
        const resp = await fetch(API.upscale, {
          method: 'POST',
          headers: apiHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            image: url,
            format: fmt,
            page_text: pageText,
            page_width: pageWidth,
            page_height: pageHeight,
            page_word_count: pageWordCount,
            is_image_primary: isImagePrimary,
            image_meta: meta || null,
          }),
        })
        if (!resp.ok) return url
        const data = await resp.json()
        return data.image || url
      } catch {
        return url
      }
    })
  )
  return results
}
