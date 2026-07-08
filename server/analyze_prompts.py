"""Shared slide-analysis prompts and JSON extraction for local and Kaggle LLM backends."""
from __future__ import annotations

import json
import re


def build_analyze_messages(
    text: str,
    is_digest: bool = False,
    page_layout: str = "plain",
) -> list[dict]:
    if page_layout == "plain" and "[HEADLINE]" in text:
        page_layout = "structured"

    noise_exclusion = (
        "Never include page numbers, running headers, footers, publication dates, "
        "captions, image labels, or any repeating metadata in any field.\n"
    )
    audience = (
        "You are narrating an educational presentation for curious students aged 14–18. "
        "Use clear, engaging language — avoid jargon unless you briefly explain it.\n"
    )

    structured_hints = ""
    if page_layout in ("structured", "qa"):
        structured_hints = (
            "The user text is a STRUCTURED page extract. Each line is tagged:\n"
            "- [HEADLINE] — main title lines; merge ALL headline lines into one natural title.\n"
            "- [SUBHEAD] — section subheading (use for subtitle when present).\n"
            "- [BODY] — main explanatory text (primary source for summary and highlights).\n"
            "- [CALLOUT] — diagram labels or side notes (use only if relevant).\n"
            "Respect reading order. Do not repeat headline fragments in subtitle or summary.\n"
        )
        if page_layout == "qa":
            structured_hints += (
                "This is a magazine Q&A / 'Did you know?' callout page. "
                "The title should be the full question in natural English "
                "(e.g. 'Is it really true that stars are twinkling?'). "
                "Use SUBHEAD lines for the section label. Answer using BODY text only.\n"
            )

    if is_digest:
        system_instructions = (
            f"{audience}"
            "The user will give you the full text of a magazine or newspaper page that contains "
            "MULTIPLE separate articles or news items.\n"
            "Identify every distinct article on the page and output a JSON object with exactly these fields:\n"
            "- \"title\": The section heading of the whole page (e.g. 'Science Updates') — under 8 words.\n"
            "- \"subtitle\": A short description like '5 stories this issue' — under 12 words.\n"
            "- \"summary\": One sentence summarising the page as a whole (20–30 words).\n"
            "- \"topics\": A JSON array where each element has:\n"
            "    - \"title\": The article headline — under 10 words.\n"
            "    - \"summary\": One vivid sentence (20–35 words) summarising that article.\n"
            "- \"narration\": A flowing narration (60–80 words) that briefly introduces each article "
            "one by one in order, with natural transitions between them.\n"
            f"{noise_exclusion}"
            "Output ONLY the raw JSON object — no markdown fences, no explanation.\n\n"
            "Example output format:\n"
            "{\"title\": \"Science Updates\", \"subtitle\": \"4 stories this issue\", "
            "\"summary\": \"This page covers breakthroughs in space, medicine, climate, and AI.\", "
            "\"topics\": [{\"title\": \"Mars Mission Milestone\", \"summary\": \"NASA's Perseverance rover collected its 20th rock sample, a major step toward returning Martian material to Earth.\"}], "
            "\"narration\": \"Welcome to Science Updates. This week we have four fascinating stories ...\"}"
        )
    else:
        title_rule = (
            "The main topic — concise, under 12 words."
            if page_layout in ("structured", "qa")
            else "The main topic — concise, under 8 words."
        )
        system_instructions = (
            f"{audience}"
            f"{structured_hints}"
            "Convert the user's text into a slide JSON object with exactly these fields:\n"
            f"- \"title\": {title_rule}\n"
            "- \"subtitle\": A compelling subheading — under 12 words.\n"
            "- \"summary\": Two clear sentences (30–50 words total) explaining the core concept.\n"
            "- \"highlights\": An array of exactly 3 bullet points — key takeaways, 12–25 words each, "
            "starting with a strong action verb or number.\n"
            "- \"supportingPoints\": An array of exactly 4 interesting supporting facts or details "
            "(15–30 words each) that deepen understanding.\n"
            "- \"narration\": An engaging narration script (50–70 words) for audio playback — "
            "written in present tense, conversational but informative, as if speaking directly to students.\n"
            f"{noise_exclusion}"
            "Output ONLY the raw JSON object — no markdown fences, no explanation.\n\n"
            "Example output format:\n"
            "{\"title\": \"How Black Holes Form\", \"subtitle\": \"Stars that collapse under their own gravity\", "
            "\"summary\": \"Black holes are regions in space where gravity is so strong that nothing, not even light, can escape. They form when massive stars exhaust their fuel and collapse.\", "
            "\"highlights\": [\"Stars 20× the Sun's mass can become black holes after a supernova explosion.\", "
            "\"The event horizon is the point of no return around a black hole.\", "
            "\"Supermassive black holes lurk at the centres of most large galaxies.\"], "
            "\"supportingPoints\": [\"The nearest known black hole, Gaia BH1, is 1,560 light-years from Earth.\", "
            "\"Time dilation near a black hole means clocks tick slower the closer you get.\", "
            "\"Stephen Hawking proposed that black holes slowly lose energy through Hawking radiation.\", "
            "\"The first image of a black hole was captured by the Event Horizon Telescope in 2019.\"], "
            "\"narration\": \"Black holes are one of the universe's most extreme objects. When a star much larger than our Sun runs out of fuel, it can no longer hold itself up against gravity. It collapses inward, triggering a massive explosion called a supernova, and leaving behind a region so dense that nothing — not even light — can escape. We call this boundary the event horizon. Scientists believe supermassive black holes, millions of times heavier than the Sun, sit at the heart of nearly every large galaxy, including our own Milky Way.\"}"
        )

    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": text},
    ]


def extract_analysis_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        raise ValueError("No JSON object found in LLM response.")
    json_str = text[first_brace:last_brace + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    fixed_str = re.sub(r'"\s*\]\s*\}', '"\n}', json_str)
    fixed_str = re.sub(r'"\s*,\s*\]\s*\}', '"\n}', fixed_str)

    open_braces = fixed_str.count("{")
    close_braces = fixed_str.count("}")
    if open_braces > close_braces:
        if fixed_str.count('"') % 2 != 0:
            fixed_str += '"'
        fixed_str += "}" * (open_braces - close_braces)

    try:
        return json.loads(fixed_str)
    except Exception as e:
        fallback_dict = {}
        for key in ["title", "subtitle", "summary", "narration"]:
            match = re.search(r'"' + key + r'"\s*:\s*"([^"]*)"', json_str)
            if match:
                fallback_dict[key] = match.group(1)

        for key in ["highlights", "supportingPoints"]:
            array_match = re.search(r'"' + key + r'"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if array_match:
                items = re.findall(r'"([^"]*)"', array_match.group(1))
                fallback_dict[key] = items

        if "title" in fallback_dict:
            fallback_dict.setdefault("highlights", [])
            fallback_dict.setdefault("supportingPoints", [])
            return fallback_dict

        raise ValueError(f"Failed to parse or repair JSON response: {e}") from e
