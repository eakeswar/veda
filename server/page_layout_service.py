"""PDF page layout extraction: text lines and images."""

from __future__ import annotations

import base64
import re

import fitz

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_IMPORTED = True
except ImportError:
    _pdfplumber = None  # type: ignore[assignment,misc]
    _PDFPLUMBER_IMPORTED = False


def reconstruct_line_text(spans) -> str:
    line_text = ""
    sorted_spans = sorted(spans, key=lambda s: s.get("origin", (0, 0))[0])
    for s in sorted_spans:
        text = s.get("text", "")
        if not text:
            continue
        if not line_text:
            line_text = text
        else:
            needs_space = not line_text.endswith('-') and not text.startswith(',') and not text.startswith('.')
            line_text += " " + text if needs_space else text
    return line_text.strip()


class PDFMetadataAnalyzer:
    def __init__(self, doc):
        self.running_headers_footers = set()
        self._analyze(doc)

    def _analyze(self, doc):
        from collections import Counter
        top_lines = []
        bottom_lines = []

        for page in doc:
            height = page.rect.height
            d = page.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type") == 0:
                    for l in b.get("lines", []):
                        spans = l.get("spans", [])
                        if not spans:
                            continue
                        line_text = reconstruct_line_text(spans)
                        if not line_text:
                            continue
                        y = spans[0]["origin"][1]
                        if y < height * 0.10:
                            top_lines.append(line_text)
                        elif y > height * 0.90:
                            bottom_lines.append(line_text)

        top_counts = Counter(top_lines)
        bottom_counts = Counter(bottom_lines)

        total_pages = len(doc)
        min_count = max(2, min(3, total_pages))

        for text, count in top_counts.items():
            if count >= min_count:
                self.running_headers_footers.add(text)

        for text, count in bottom_counts.items():
            if count >= min_count:
                self.running_headers_footers.add(text)


def _extract_lines_pymupdf(page_obj, width, height, active_analyzer):
    """
    PyMuPDF-based text line extraction — used as fallback when pdfplumber is
    unavailable or fails.  Applies the same header/footer/margin filters as the
    pdfplumber path.
    """
    d = page_obj.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            if not spans:
                continue
            line_text = reconstruct_line_text(spans)
            if not line_text:
                continue
            if active_analyzer and line_text in active_analyzer.running_headers_footers:
                continue
            origin_y = spans[0]["origin"][1]
            is_margin = (origin_y < height * 0.10) or (origin_y > height * 0.90)
            if is_margin:
                if re.match(r'^\d+$', line_text) or re.match(r'^[ivxIVX]+$', line_text):
                    continue
                if re.match(r'^(page|slide|p\.)\s*\d+$', line_text, re.IGNORECASE):
                    continue
            max_font_size = 0.0
            min_x = float("inf")
            baseline_y = 0.0
            for s in spans:
                max_font_size = max(max_font_size, s["size"])
                min_x = min(min_x, s["origin"][0])
                baseline_y = max(baseline_y, height - s["origin"][1])
            lines.append({
                "text": line_text,
                "x": min_x,
                "y": baseline_y,
                "fontSize": max_font_size,
            })
    return lines


def _words_overlapping_image_count(words, ix0, iy0, ix1, iy1) -> int:
    """Count pdfplumber words overlapping an image rect (>45% word width)."""
    count = 0
    for w in words:
        wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]
        ox = min(wx1, ix1) - max(wx0, ix0)
        oy = min(wy1, iy1) - max(wy0, iy0)
        if ox > 0 and oy > 0:
            word_w = wx1 - wx0 or 1
            if ox / word_w > 0.45:
                count += 1
    return count


def _attach_overlap_word_counts(
    images: list[dict],
    pdf_path: str,
    page_num: int,
    page_height: float,
    pdfplumber_available: bool,
) -> None:
    """Set overlapWordCount on each image dict (words overlapping image zone)."""
    for img in images:
        img["overlapWordCount"] = 0

    if not images or not pdfplumber_available or not _PDFPLUMBER_IMPORTED:
        return

    try:
        with _pdfplumber.open(pdf_path) as pdf:
            pl_page = pdf.pages[page_num - 1]
            try:
                words = pl_page.extract_words(
                    keep_blank_chars=False,
                    x_tolerance=3,
                    y_tolerance=3,
                )
            except Exception:
                words = pl_page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)

        for img in images:
            td = (
                img["x"],
                page_height - img["y"] - img["h"],
                img["x"] + img["w"],
                page_height - img["y"],
            )
            img["overlapWordCount"] = _words_overlapping_image_count(words, *td)
    except Exception as exc:
        print(f"overlapWordCount skipped for page {page_num} ({exc})")


def _content_image_rects(img_rects_topdown, width, height):
    """Image rects that should suppress overlapping text (exclude full-page backgrounds)."""
    page_area = width * height or 1
    content = []
    for x0, y0, x1, y1 in img_rects_topdown:
        w = x1 - x0
        h = y1 - y0
        if w > width * 0.85 or h > height * 0.85:
            continue
        if w * h > page_area * 0.45:
            continue
        content.append((x0, y0, x1, y1))
    return content


def _group_words_into_lines(words):
    """Group pdfplumber word dicts into lines by proximity of their `top` coordinate."""
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur = [], [sorted_w[0]]
    cur_top = sorted_w[0]["top"]
    for w in sorted_w[1:]:
        if abs(w["top"] - cur_top) <= 4:
            cur.append(w)
        else:
            lines.append(sorted(cur, key=lambda w: w["x0"]))
            cur, cur_top = [w], w["top"]
    lines.append(sorted(cur, key=lambda w: w["x0"]))
    return lines


def _extract_lines_pdfplumber(pdf_path, page_num, img_rects_topdown, width, height, active_analyzer):
    """
    Layout-aware text extraction combining pdfplumber (text) and pre-extracted image
    rectangles (to filter text that bleeds over image zones).

    img_rects_topdown: list of (x0, y0, x1, y1) in top-down screen coords.

    Returns list of {"text", "x", "y" (bottom-up), "fontSize"} to match the
    existing PyMuPDF output format expected by the frontend.

    Column-detection approach:
    - Sample horizontal coverage across the middle 40 % of the page.
    - A gap ≥ 3 % of page width in that region signals a column gutter.
    - Full-width spans (crossing the gutter) are treated as headers/titles and
      placed before/after column content depending on their vertical position.
    """
    with _pdfplumber.open(pdf_path) as pdf:
        pl_page = pdf.pages[page_num - 1]

        # Words with per-word font size (falls back gracefully if attr missing)
        try:
            words = pl_page.extract_words(
                extra_attrs=["size"],
                keep_blank_chars=False,
                x_tolerance=3,
                y_tolerance=3,
            )
        except Exception:
            words = pl_page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)
            for w in words:
                w.setdefault("size", 12.0)

        if not words:
            return []

        # Filter words overlapping inset photos/diagrams — not decorative backgrounds.
        filter_rects = _content_image_rects(img_rects_topdown, width, height)

        def _overlaps_image(w):
            wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]
            for (ix0, iy0, ix1, iy1) in filter_rects:
                ox = min(wx1, ix1) - max(wx0, ix0)
                oy = min(wy1, iy1) - max(wy0, iy0)
                if ox > 0 and oy > 0:
                    word_w = wx1 - wx0 or 1
                    if ox / word_w > 0.45:
                        return True
            return False

        filtered = [w for w in words if not _overlaps_image(w)]
        if not filtered:
            return []

        # ── Column detection ─────────────────────────────────────────────────
        mid_start = int(width * 0.30)
        mid_end = int(width * 0.70)
        x_spans = [(w["x0"], w["x1"]) for w in filtered]

        # Count how many words cover each x position (sampled every 2 pts)
        coverage = {x: sum(1 for (x0, x1) in x_spans if x0 <= x <= x1)
                    for x in range(mid_start, mid_end, 2)}

        # Find the widest contiguous zero-coverage gap in the middle band
        gutter_x = None
        max_gap = 0
        gap_start = None
        for x in range(mid_start, mid_end, 2):
            if coverage.get(x, 0) == 0:
                if gap_start is None:
                    gap_start = x
            else:
                if gap_start is not None:
                    gap = x - gap_start
                    if gap > max_gap:
                        max_gap, gutter_x = gap, (gap_start + x) // 2
                    gap_start = None
        if gap_start is not None:
            gap = mid_end - gap_start
            if gap > max_gap:
                max_gap, gutter_x = gap, (gap_start + mid_end) // 2

        is_two_col = max_gap >= width * 0.03 and gutter_x is not None

        # ── Group into lines then classify ───────────────────────────────────
        all_line_groups = _group_words_into_lines(filtered)

        if is_two_col:
            # A line "spans the gutter" if it has words on both sides
            def _spans(line):
                return any(w["x0"] < gutter_x for w in line) and any(w["x1"] > gutter_x for w in line)

            full_lines   = [l for l in all_line_groups if _spans(l)]
            left_lines   = [l for l in all_line_groups if not _spans(l) and all(w["x1"] <= gutter_x for w in l)]
            right_lines  = [l for l in all_line_groups if not _spans(l) and all(w["x0"] >= gutter_x for w in l)]

            # Column content starts where the first left or right column word appears
            col_words = [w for l in left_lines + right_lines for w in l]
            col_start = min((w["top"] for w in col_words), default=0)
            col_end   = max((w["top"] for w in col_words), default=height)

            pre_col  = sorted([l for l in full_lines if l[0]["top"] < col_start],  key=lambda l: l[0]["top"])
            post_col = sorted([l for l in full_lines if l[0]["top"] > col_end],   key=lambda l: l[0]["top"])
            mid_full = sorted([l for l in full_lines if col_start <= l[0]["top"] <= col_end], key=lambda l: l[0]["top"])

            # Primary column = the one with more words (main article body).
            # Secondary column = the one with fewer words (sidebar / captions).
            # If counts are within 1.8× of each other, use left-first (standard order).
            left_wc  = sum(len(l) for l in left_lines)
            right_wc = sum(len(l) for l in right_lines)
            if right_wc > left_wc * 1.8:
                primary_col   = sorted(right_lines, key=lambda l: l[0]["top"])
                secondary_col = sorted(left_lines,  key=lambda l: l[0]["top"])
            else:
                primary_col   = sorted(left_lines,  key=lambda l: l[0]["top"])
                secondary_col = sorted(right_lines, key=lambda l: l[0]["top"])

            ordered = (
                pre_col
                + primary_col
                + mid_full
                + secondary_col
                + post_col
            )
        else:
            ordered = all_line_groups

        # ── Build output, applying header/footer filters ─────────────────────
        result = []
        for line_words in ordered:
            text = " ".join(w["text"] for w in line_words).strip()
            if not text:
                continue

            if active_analyzer and text in active_analyzer.running_headers_footers:
                continue

            top_y = line_words[0]["top"]
            is_margin = top_y < height * 0.10 or top_y > height * 0.90
            if is_margin:
                if re.match(r'^\d+$', text) or re.match(r'^[ivxIVX]+$', text):
                    continue
                if re.match(r'^(page|slide|p\.)\s*\d+$', text, re.IGNORECASE):
                    continue

            sizes = [w.get("size") or 12.0 for w in line_words]
            font_size = max(sizes)
            min_x = min(w["x0"] for w in line_words)
            # Convert top-down bottom coord → bottom-up y (matches PyMuPDF output)
            y_bottom_up = height - line_words[0]["bottom"]

            result.append({
                "text": text,
                "x": min_x,
                "y": y_bottom_up,
                "fontSize": font_size,
            })

        return result


def extract_page_layout(
    active_doc,
    active_analyzer,
    page: int,
    pdfplumber_available: bool = True,
) -> dict:
    """Extract {width, height, lines, images} for a single PDF page."""
    page_obj = active_doc[page - 1]
    rect = page_obj.rect
    width = rect.width
    height = rect.height

    lines = []
    images = []

    # ── Image extraction via get_images() ────────────────────────────────
    # get_text("dict") type-1 blocks miss most PDF images because the
    # majority of PDFs embed images as XObjects (placed with Do operator)
    # rather than inline. get_images(full=True) finds all of them.
    seen_xrefs = set()
    for img_info in page_obj.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            rects = page_obj.get_image_rects(xref)
        except Exception:
            rects = []

        for rect in rects:
            img_w = rect.width
            img_h = rect.height
            # Skip invisible or near-zero rendered size
            if img_w < 5 or img_h < 5:
                continue
            # Skip tiny decorative elements (icons, bullets, dividers)
            if img_w < 50 or img_h < 50:
                continue
            # Skip full-page-width backgrounds / decorative banners
            if img_w > width * 0.90:
                continue

            try:
                img_dict = active_doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = img_dict.get("image")
            if not image_bytes:
                continue
            ext = img_dict.get("ext", "png")
            img_base64 = base64.b64encode(image_bytes).decode("utf-8")
            url = f"data:image/{ext};base64,{img_base64}"
            images.append({
                "url": url,
                "x": rect.x0,
                "y": height - rect.y1,
                "w": img_w,
                "h": img_h,
                "cx": (rect.x0 + rect.x1) / 2,
                "cy": height - (rect.y0 + rect.y1) / 2
            })
            break  # one rect per xref is enough for position info

    _attach_overlap_word_counts(
        images,
        str(active_doc.name),
        page,
        height,
        pdfplumber_available,
    )

    # ── Text extraction ───────────────────────────────────────────────
    # pdfplumber gives column-aware, image-zone-filtered text.
    # Fall back to PyMuPDF's get_text("dict") if pdfplumber is unavailable
    # or raises an unexpected error.
    use_pdfplumber = pdfplumber_available and _PDFPLUMBER_IMPORTED
    if use_pdfplumber:
        try:
            # Convert image positions to top-down coords for overlap filtering
            img_rects_td = [
                (img["x"], height - img["y"] - img["h"],
                 img["x"] + img["w"], height - img["y"])
                for img in images
            ]
            lines = _extract_lines_pdfplumber(
                str(active_doc.name), page, img_rects_td,
                width, height, active_analyzer
            )
            print(f"pdfplumber extracted {len(lines)} lines for page {page}")
        except Exception as pl_err:
            print(f"pdfplumber failed for page {page} ({pl_err}), falling back to PyMuPDF")
            lines = _extract_lines_pymupdf(page_obj, width, height, active_analyzer)
    else:
        lines = _extract_lines_pymupdf(page_obj, width, height, active_analyzer)

    return {
        "width": width,
        "height": height,
        "lines": lines,
        "images": images
    }
