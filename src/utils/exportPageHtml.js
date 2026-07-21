/** Build a self-contained HTML file for one Veda slide (trial export). */

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function slugify(text, pageNumber) {
  const slug = String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 48)
  return `veda-page-${String(pageNumber).padStart(2, '0')}${slug ? `-${slug}` : ''}.html`
}

function getTopicSpeechText(topic, language) {
  if (!topic) return ''
  if (language === 'te-IN') {
    return (
      topic.narration_te
      || `${topic.title || ''}. ${topic.summary || topic.body || ''}`.replace(/\s+/g, ' ').trim()
    )
  }
  return `${topic.title || ''}. ${topic.summary || topic.body || ''}`.replace(/\s+/g, ' ').trim()
}

function getDeckSpeechText(deck, language, pageText) {
  if (!deck) return pageText || ''
  if (language === 'te-IN') {
    return deck.narration_te || deck.narration || pageText || ''
  }
  return deck.narration || pageText || ''
}

async function blobUrlToDataUrl(url) {
  if (!url || !url.startsWith('blob:')) {
    return url || null
  }
  const response = await fetch(url)
  const blob = await response.blob()
  return await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

function excerptText(text, maxLen = 1400) {
  const trimmed = String(text || '').replace(/\s+/g, ' ').trim()
  if (!trimmed) return ''
  if (trimmed.length <= maxLen) return trimmed
  return `${trimmed.slice(0, maxLen).replace(/\s+\S*$/, '')}…`
}

function splitIntoParagraphs(text, sentencesPerPara = 3) {
  const sentences = String(text || '').split(/(?<=[.!?])\s+/).filter((s) => s.trim().length > 0)
  if (sentences.length === 0) return []
  const paragraphs = []
  for (let i = 0; i < sentences.length; i += sentencesPerPara) {
    paragraphs.push(sentences.slice(i, i + sentencesPerPara).join(' '))
  }
  return paragraphs
}

function buildArticleBody(deck, language, sourceText) {
  const narration = getDeckSpeechText(deck, language, sourceText)
  if (narration && narration.length > 40) return narration
  if (deck.summary && deck.summary.length > 20) return deck.summary
  return excerptText(sourceText || deck.sourceText || '')
}

function uniqueImages(deck) {
  const urls = [
    ...(deck.images || []),
    ...(deck.topics || []).map((topic) => topic.image).filter(Boolean),
  ]
  return [...new Set(urls.filter(Boolean))]
}

export async function buildPageExportPayload(deck, options = {}) {
  const {
    pageNumber = 1,
    pageLabel = '',
    language = 'en-US',
    theme = 'light',
    sourceText = '',
    cachedAudio = null,
  } = options

  if (!deck) {
    throw new Error('No slide data to export')
  }

  let audioDataUrl = null
  let wordBoundaries = []
  if (cachedAudio?.url) {
    audioDataUrl = await blobUrlToDataUrl(cachedAudio.url)
    wordBoundaries = cachedAudio.wordBoundaries || []
  }

  const images = uniqueImages(deck)
  const isDigest = !!deck.isDigest && (deck.topics || []).length > 0

  let stories = []
  if (isDigest) {
    stories = (deck.topics || []).map((topic, index) => {
      const narration = getTopicSpeechText(topic, language)
      const hasCachedAudio =
        cachedAudio
        && cachedAudio.text === narration
        && cachedAudio.language === language

      return {
        index: index + 1,
        title: topic.title || `Story ${index + 1}`,
        body: topic.body || topic.summary || '',
        image: topic.image || null,
        narration,
        audioDataUrl: hasCachedAudio ? audioDataUrl : null,
        wordBoundaries: hasCachedAudio ? wordBoundaries : [],
      }
    })
  } else {
    const narration = getDeckSpeechText(deck, language, sourceText)
    const articleBody = buildArticleBody(deck, language, sourceText)
    stories = [{
      index: 1,
      title: deck.title || pageLabel || `Page ${pageNumber}`,
      body: articleBody,
      summary: deck.summary || '',
      image: (deck.images || [])[0] || null,
      narration,
      audioDataUrl,
      wordBoundaries,
    }]
  }

  return {
    meta: {
      pageNumber,
      pageLabel: pageLabel || deck.title || `Page ${pageNumber}`,
      language,
      theme,
      exportedAt: new Date().toISOString(),
      isDigest,
      isImagePrimary: !!deck.isImagePrimary,
    },
    deck: {
      title: deck.title || pageLabel || `Page ${pageNumber}`,
      subtitle: deck.subtitle || '',
      summary: deck.summary || '',
      highlights: deck.highlights || [],
      supportingPoints: deck.supportingPoints || [],
      sourceExcerpt: excerptText(sourceText || deck.sourceText || ''),
      images,
    },
    stories,
  }
}

function buildExportHtml(payload) {
  const langClass = payload.meta.language === 'te-IN' ? 'lang-te' : ''
  const dataJson = JSON.stringify(payload).replace(/</g, '\\u003c')
  const kicker = payload.meta.isDigest
    ? `Page ${payload.meta.pageNumber} · ${payload.stories.length} stories`
    : `Page ${payload.meta.pageNumber}`

  return `<!DOCTYPE html>
<html lang="${payload.meta.language === 'te-IN' ? 'te' : 'en'}" data-theme="${escapeHtml(payload.meta.theme)}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${escapeHtml(payload.deck.title)} · Veda</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+Telugu:wght@400;600&family=Lora:ital,wght@0,400;0,600;1,400&family=Outfit:wght@600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --accent: #d97706;
      --accent-soft: rgba(217, 119, 6, 0.12);
      --bg: #ffffff;
      --text: #1a1a1a;
      --text-secondary: #5c5c5c;
      --text-tertiary: #8a8a8a;
      --line: #ececec;
      --measure: 42rem;
      --measure-wide: 56rem;
    }
    html[data-theme="dark"] {
      --accent: #f5a623;
      --accent-soft: rgba(245, 166, 35, 0.14);
      --bg: #111111;
      --text: #f2efe8;
      --text-secondary: #b8b0a4;
      --text-tertiary: #8a8278;
      --line: #2a2a2a;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100vh; }
    body {
      font-family: Inter, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      -webkit-font-smoothing: antialiased;
    }
    .lang-te { font-family: "Noto Sans Telugu", Inter, system-ui, sans-serif; }
    .site-nav {
      position: sticky; top: 0; z-index: 50;
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px clamp(20px, 5vw, 48px);
      background: color-mix(in srgb, var(--bg) 88%, transparent);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--line);
    }
    .site-nav .logo {
      font-family: Outfit, sans-serif;
      font-weight: 700; font-size: 0.9rem; letter-spacing: -0.02em;
      color: var(--text);
    }
    .site-nav .logo em { font-style: normal; color: var(--accent); }
    .nav-actions { display: flex; gap: 10px; align-items: center; }
    .nav-meta { font-size: 0.75rem; color: var(--text-tertiary); font-weight: 500; }
    .btn-ghost {
      border: 1px solid var(--line); background: transparent; color: var(--text-secondary);
      width: 34px; height: 34px; border-radius: 50%; cursor: pointer; font-size: 0.85rem;
    }
    .article {
      max-width: var(--measure-wide);
      margin: 0 auto;
      padding: clamp(32px, 6vw, 64px) clamp(20px, 5vw, 48px) 80px;
    }
    .article-header {
      max-width: var(--measure);
      margin: 0 auto 40px;
      text-align: center;
    }
    .kicker {
      margin: 0 0 16px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }
    .headline {
      margin: 0;
      font-family: Outfit, Georgia, serif;
      font-size: clamp(2rem, 5vw, 3rem);
      font-weight: 700;
      line-height: 1.12;
      letter-spacing: -0.03em;
    }
    .dek {
      margin: 18px auto 0;
      max-width: 36rem;
      font-family: Lora, Georgia, serif;
      font-size: clamp(1.05rem, 2.2vw, 1.35rem);
      line-height: 1.55;
      color: var(--text-secondary);
      font-weight: 400;
    }
    .lede {
      margin: 24px auto 0;
      max-width: 38rem;
      font-size: 1.05rem;
      line-height: 1.75;
      color: var(--text-secondary);
    }
    .hero-figure {
      margin: 0 auto 48px;
      max-width: var(--measure-wide);
    }
    .hero-figure img {
      width: 100%;
      max-height: min(520px, 62vh);
      object-fit: contain;
      display: block;
      margin: 0 auto;
      border-radius: 4px;
      cursor: zoom-in;
    }
    .hero-thumbs {
      display: flex; gap: 8px; justify-content: center;
      flex-wrap: wrap; margin-top: 12px;
    }
    .hero-thumbs img {
      width: 56px; height: 56px; object-fit: cover; border-radius: 4px;
      opacity: 0.55; cursor: pointer; transition: opacity 0.15s;
    }
    .hero-thumbs img.active, .hero-thumbs img:hover { opacity: 1; }
    .gallery {
      display: flex; gap: 12px; overflow-x: auto;
      margin: 0 auto 48px; max-width: var(--measure-wide);
      padding-bottom: 8px; scrollbar-width: thin;
    }
    .gallery img {
      height: 160px; width: auto; border-radius: 4px;
      object-fit: cover; cursor: zoom-in; flex-shrink: 0;
    }
    .content-flow {
      max-width: var(--measure);
      margin: 0 auto;
    }
    .story-section {
      margin-bottom: 56px;
      padding-bottom: 56px;
      border-bottom: 1px solid var(--line);
    }
    .story-section:last-child { border-bottom: none; padding-bottom: 0; }
    .story-label {
      margin: 0 0 12px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--text-tertiary);
    }
    .story-heading {
      margin: 0 0 20px;
      font-family: Outfit, sans-serif;
      font-size: clamp(1.4rem, 3vw, 1.85rem);
      font-weight: 700;
      line-height: 1.2;
      letter-spacing: -0.02em;
    }
    .story-figure { margin: 0 0 24px; }
    .story-figure img {
      width: 100%; max-height: 400px; object-fit: cover;
      border-radius: 4px; cursor: zoom-in; display: block;
    }
    .story-text {
      margin: 0 0 28px;
      font-family: Lora, Georgia, serif;
      font-size: 1.125rem;
      line-height: 1.8;
      color: var(--text);
    }
    .article-body {
      margin: 0 0 36px;
    }
    .article-body p {
      margin: 0 0 1.25em;
      font-family: Lora, Georgia, serif;
      font-size: 1.125rem;
      line-height: 1.85;
      color: var(--text);
    }
    .article-body p:last-child { margin-bottom: 0; }
    .reading-block {
      margin: 36px 0;
      padding-top: 8px;
    }
    .reading-block .block-label { margin-bottom: 16px; }
    .glance-list, .detail-list {
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .glance-list li {
      position: relative;
      padding: 0 0 16px 18px;
      margin: 0 0 16px;
      border-left: 3px solid var(--accent);
      font-family: Lora, Georgia, serif;
      font-size: 1.05rem;
      line-height: 1.7;
      color: var(--text-secondary);
    }
    .glance-list li:last-child { margin-bottom: 0; padding-bottom: 0; }
    .detail-list li {
      padding: 0 0 14px;
      margin: 0 0 14px;
      border-bottom: 1px solid var(--line);
      font-size: 1rem;
      line-height: 1.7;
      color: var(--text-secondary);
    }
    .detail-list li:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
    .digest-intro {
      margin: 0 auto 40px;
      max-width: var(--measure);
      font-family: Lora, Georgia, serif;
      font-size: 1.125rem;
      line-height: 1.8;
      color: var(--text-secondary);
      text-align: center;
    }
    .audio-block {
      background: color-mix(in srgb, var(--bg) 92%, var(--accent) 8%);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px 22px;
      margin: 32px 0;
    }
    .audio-block.compact { margin: 28px 0 0; }
    .audio-top {
      display: flex; align-items: center; gap: 14px; margin-bottom: 16px;
    }
    .btn-listen {
      flex-shrink: 0;
      border: none;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      font-size: 0.85rem;
      padding: 10px 18px;
      border-radius: 999px;
      cursor: pointer;
    }
    html[data-theme="dark"] .btn-listen { color: #111; }
    .btn-listen[disabled] { opacity: 0.4; cursor: not-allowed; }
    .audio-label {
      font-size: 0.8rem;
      color: var(--text-tertiary);
      line-height: 1.4;
    }
    .karaoke {
      font-family: Lora, Georgia, serif;
      font-size: 1.05rem;
      line-height: 1.85;
      color: var(--text-secondary);
    }
    .karaoke .word {
      padding: 1px 0;
      border-radius: 2px;
      transition: color 0.12s, background 0.12s;
    }
    .karaoke .word.active {
      color: var(--text);
      background: var(--accent-soft);
      font-weight: 600;
    }
    .takeaways-block {
      margin: 56px auto 0;
      max-width: var(--measure);
      padding: 28px 0 0;
      border-top: 1px solid var(--line);
    }
    .block-label {
      margin: 0 0 20px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--text-tertiary);
    }
    .takeaway-list { margin: 0; padding: 0; list-style: none; }
    .takeaway-list li {
      position: relative;
      padding: 0 0 20px 20px;
      margin: 0 0 20px;
      border-left: 3px solid var(--accent);
      font-size: 1.05rem;
      line-height: 1.65;
      color: var(--text-secondary);
    }
    .takeaway-list li:last-child { margin-bottom: 0; padding-bottom: 0; }
    .notes-block {
      margin: 48px auto 0;
      max-width: var(--measure);
      padding-top: 28px;
      border-top: 1px solid var(--line);
    }
    .notes-list {
      margin: 0; padding: 0; list-style: none;
    }
    .notes-list li {
      padding: 14px 0;
      border-bottom: 1px solid var(--line);
      font-size: 0.95rem;
      line-height: 1.65;
      color: var(--text-secondary);
    }
    .notes-list li:last-child { border-bottom: none; }
    .article-end {
      margin: 64px auto 0;
      max-width: var(--measure);
      text-align: center;
      font-size: 0.75rem;
      color: var(--text-tertiary);
    }
    .lightbox {
      position: fixed; inset: 0; z-index: 1000;
      background: rgba(0,0,0,0.9);
      display: none; pointer-events: none; visibility: hidden;
      align-items: center; justify-content: center; padding: 24px;
    }
    .lightbox.open { display: flex; pointer-events: auto; visibility: visible; }
    .lightbox img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .lightbox button {
      position: absolute; top: 20px; right: 20px;
      width: 40px; height: 40px; border-radius: 50%;
      border: none; background: #fff; cursor: pointer; font-size: 1.1rem;
    }
    @media (max-width: 640px) {
      .article-header { text-align: left; }
      .hero-thumbs { justify-content: flex-start; }
    }
  </style>
</head>
<body class="${langClass}">
  <nav class="site-nav">
    <div class="logo">Veda <em>Reader</em></div>
    <div class="nav-actions">
      <span class="nav-meta">${escapeHtml(kicker)}</span>
      <button class="btn-ghost" id="theme-toggle" title="Toggle theme">🌙</button>
    </div>
  </nav>

  <article class="article">
    <header class="article-header">
      <p class="kicker">${payload.meta.isDigest ? 'Digest' : 'Article'}</p>
      <h1 class="headline">${escapeHtml(payload.deck.title)}</h1>
      ${payload.deck.subtitle ? `<p class="dek">${escapeHtml(payload.deck.subtitle)}</p>` : ''}
      ${payload.deck.summary && !payload.meta.isDigest ? `<p class="lede">${escapeHtml(payload.deck.summary)}</p>` : ''}
    </header>

    ${payload.meta.isDigest && payload.deck.summary ? `
      <p class="digest-intro">${escapeHtml(payload.deck.summary)}</p>
    ` : ''}

    ${!payload.meta.isDigest && payload.deck.images.length ? `
      <figure class="hero-figure">
        <img id="hero-main" alt="${escapeHtml(payload.deck.title)}" />
        ${payload.deck.images.length > 1 ? '<div class="hero-thumbs" id="hero-thumbs"></div>' : ''}
      </figure>
    ` : ''}

    ${payload.meta.isDigest && payload.deck.images.length ? `<div class="gallery" id="image-strip"></div>` : ''}

    <div class="content-flow">
      <div id="stories"></div>
      <div id="reading-sections"></div>
      <p class="article-end">Presented by Veda</p>
    </div>
  </article>

  <div class="lightbox" id="lightbox" aria-hidden="true">
    <button id="lightbox-close" title="Close">✕</button>
    <img id="lightbox-img" alt="" />
  </div>

  <script type="application/json" id="veda-export-data">${dataJson}</script>
  <script>
    const payload = JSON.parse(document.getElementById('veda-export-data').textContent);
    const themeBtn = document.getElementById('theme-toggle');

    function setTheme(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
    setTheme(payload.meta.theme || 'light');
    themeBtn.addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      setTheme(next);
    });

    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text != null) node.textContent = text;
      return node;
    }

    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    function openLightbox(src) {
      lightboxImg.src = src;
      lightbox.classList.add('open');
    }
    document.getElementById('lightbox-close').addEventListener('click', () => lightbox.classList.remove('open'));
    lightbox.addEventListener('click', (e) => { if (e.target === lightbox) lightbox.classList.remove('open'); });

    const strip = document.getElementById('image-strip');
    if (strip) {
      payload.deck.images.forEach((src) => {
        const img = document.createElement('img');
        img.src = src;
        img.alt = 'Gallery image';
        img.addEventListener('click', () => openLightbox(src));
        strip.appendChild(img);
      });
    }

    const heroMain = document.getElementById('hero-main');
    const heroThumbs = document.getElementById('hero-thumbs');
    if (heroMain && payload.deck.images.length) {
      let heroIndex = 0;
      const setHero = (index) => {
        heroIndex = index;
        heroMain.src = payload.deck.images[index];
        if (heroThumbs) {
          heroThumbs.querySelectorAll('img').forEach((node, idx) => {
            node.classList.toggle('active', idx === index);
          });
        }
      };
      setHero(0);
      heroMain.addEventListener('click', () => openLightbox(payload.deck.images[heroIndex]));
      if (heroThumbs) {
        payload.deck.images.forEach((src, index) => {
          const thumb = document.createElement('img');
          thumb.src = src;
          thumb.alt = 'Thumbnail ' + (index + 1);
          if (index === 0) thumb.classList.add('active');
          thumb.addEventListener('click', () => setHero(index));
          heroThumbs.appendChild(thumb);
        });
      }
    }

    const readingSections = document.getElementById('reading-sections');

    function splitIntoParagraphs(text, sentencesPerPara) {
      const sentences = String(text || '').split(/(?<=[.!?])\\s+/).filter((s) => s.trim().length > 0);
      if (!sentences.length) return [];
      const paragraphs = [];
      for (let i = 0; i < sentences.length; i += sentencesPerPara) {
        paragraphs.push(sentences.slice(i, i + sentencesPerPara).join(' '));
      }
      return paragraphs;
    }

    function appendParagraphs(container, text) {
      splitIntoParagraphs(text, 3).forEach((paragraph) => {
        container.appendChild(el('p', '', paragraph));
      });
    }

    function buildListSection(label, listClass, items) {
      if (!items?.length) return null;
      const section = el('section', 'reading-block');
      section.appendChild(el('h2', 'block-label', label));
      const list = el('ul', listClass);
      items.forEach((item) => list.appendChild(el('li', '', item)));
      section.appendChild(list);
      return section;
    }

    let activeAudio = null;
    let activeStoryId = null;

    function splitWords(text) {
      return String(text || '').trim().split(/\\s+/).filter(Boolean);
    }

    function renderKaraoke(container, text, boundaries) {
      container.innerHTML = '';
      const words = splitWords(text);
      if (!boundaries?.length) {
        words.forEach((word) => container.appendChild(el('span', 'word', word + ' ')));
        return;
      }
      boundaries.forEach((wb, idx) => {
        const span = el('span', 'word', (wb.text || words[idx] || '') + ' ');
        span.dataset.start = wb.start;
        span.dataset.end = wb.end;
        container.appendChild(span);
      });
    }

    function highlightAt(container, t) {
      container.querySelectorAll('.word').forEach((node) => {
        const start = Number(node.dataset.start || 0);
        const end = Number(node.dataset.end || 0);
        node.classList.toggle('active', t >= start && t < end);
      });
    }

    function stopPlayback() {
      if (activeAudio) { activeAudio.pause(); activeAudio = null; }
      activeStoryId = null;
      document.querySelectorAll('.btn-listen').forEach((btn) => { btn.textContent = 'Listen'; });
    }

    function playStory(story, btn, karaokeEl) {
      if (!story.audioDataUrl) return;
      if (activeStoryId === story.index && activeAudio && !activeAudio.paused) {
        stopPlayback();
        return;
      }
      stopPlayback();
      activeStoryId = story.index;
      btn.textContent = 'Pause';
      activeAudio = new Audio(story.audioDataUrl);
      renderKaraoke(karaokeEl, story.narration, story.wordBoundaries);
      activeAudio.addEventListener('timeupdate', () => highlightAt(karaokeEl, activeAudio.currentTime));
      activeAudio.addEventListener('ended', stopPlayback);
      activeAudio.play().catch(() => stopPlayback());
    }

    function buildAudioBlock(story, compact) {
      const block = el('div', 'audio-block' + (compact ? ' compact' : ''));
      const top = el('div', 'audio-top');
      const btn = el('button', 'btn-listen', 'Listen');
      const label = el('div', 'audio-label', story.audioDataUrl
        ? 'Narrated presentation · tap to play'
        : 'Play narration in Veda first to include audio here');
      const karaoke = el('div', 'karaoke');
      if (story.audioDataUrl) {
        btn.addEventListener('click', () => playStory(story, btn, karaoke));
      } else {
        btn.disabled = true;
      }
      top.appendChild(btn);
      top.appendChild(label);
      block.appendChild(top);
      renderKaraoke(karaoke, story.narration, story.wordBoundaries);
      block.appendChild(karaoke);
      return block;
    }

    const storiesRoot = document.getElementById('stories');

    if (!payload.meta.isDigest) {
      const story = payload.stories[0];
      if (story) {
        if (story.body) {
          const bodySection = el('section', 'article-body');
          appendParagraphs(bodySection, story.body);
          storiesRoot.appendChild(bodySection);
        }

        const glance = buildListSection('At a glance', 'glance-list', payload.deck.highlights);
        if (glance) storiesRoot.appendChild(glance);

        storiesRoot.appendChild(buildAudioBlock(story, false));

        const details = buildListSection('More from this page', 'detail-list', payload.deck.supportingPoints);
        if (details) storiesRoot.appendChild(details);

        if (payload.deck.sourceExcerpt && payload.deck.sourceExcerpt.length > 80
          && payload.deck.sourceExcerpt !== story.body) {
          const excerptBlock = buildListSection('From the original page', 'detail-list', [payload.deck.sourceExcerpt]);
          if (excerptBlock) storiesRoot.appendChild(excerptBlock);
        }
      }
    } else {
      payload.stories.forEach((story) => {
        const section = el('section', 'story-section');
        section.appendChild(el('p', 'story-label', 'Story ' + story.index));
        section.appendChild(el('h2', 'story-heading', story.title));
        if (story.image) {
          const fig = el('figure', 'story-figure');
          const img = document.createElement('img');
          img.src = story.image;
          img.alt = story.title;
          img.addEventListener('click', () => openLightbox(story.image));
          fig.appendChild(img);
          section.appendChild(fig);
        }
        if (story.body) {
          const body = el('div', 'story-text');
          appendParagraphs(body, story.body);
          section.appendChild(body);
        }
        section.appendChild(buildAudioBlock(story, true));
        storiesRoot.appendChild(section);
      });

      const digestGlance = buildListSection('Key takeaways', 'glance-list', payload.deck.highlights);
      if (digestGlance) readingSections.appendChild(digestGlance);

      const digestNotes = buildListSection('Notes', 'detail-list', payload.deck.supportingPoints);
      if (digestNotes) readingSections.appendChild(digestNotes);
    }
  </script>
</body>
</html>`
}

export async function openPageHtml(deck, options = {}) {
  const payload = await buildPageExportPayload(deck, options)
  const html = buildExportHtml(payload)
  const tab = window.open('', '_blank')

  if (!tab) {
    throw new Error('Pop-up blocked — allow pop-ups for this site to open the HTML preview')
  }

  tab.document.open()
  tab.document.write(html)
  tab.document.close()
  if (payload.deck.title) {
    tab.document.title = `${payload.deck.title} · Veda`
  }

  return payload.meta
}

/** @deprecated Use openPageHtml — kept for optional file download later */
export async function downloadPageHtml(deck, options = {}) {
  const payload = await buildPageExportPayload(deck, options)
  const html = buildExportHtml(payload)
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = slugify(payload.deck.title, payload.meta.pageNumber)
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  return payload.meta
}
