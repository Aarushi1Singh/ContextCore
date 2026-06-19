"""
core/document_input.py
-----------------------
Extracts plain text from a pasted URL, so it can be fed into the same agent
pipeline that handles a typed headline.

Currently supported:
  - URL  -> fetch the page, strip HTML, extract main text, cap at
            MAX_INPUT_CHARS (same cap as the typed text area, for
            consistent cost/latency regardless of input mode).

.txt and .pdf upload are planned but not yet wired in -- PDFs in particular
need chunked extraction + entity merging for documents longer than a couple
pages, which is a separate piece of work from this simple URL fetch.
"""

import requests

MAX_INPUT_CHARS = 2000


def extract_text_from_url(url: str) -> dict:
    """
    Fetch a URL and extract readable text, stripping HTML tags/scripts.
    Returns {"text": str, "original_length": int, "truncated": bool} so the
    UI can show the user how much of the article was actually used.
    """
    response = requests.get(
        url,
        headers={"User-Agent": "ContextCore/1.0 (research tool)"},
        timeout=10,
    )
    response.raise_for_status()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements before extracting text
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = "\n".join(lines)

    return {
        "text": cleaned[:MAX_INPUT_CHARS],
        "original_length": len(cleaned),
        "truncated": len(cleaned) > MAX_INPUT_CHARS,
    }