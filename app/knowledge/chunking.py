"""Markdown document -> retrieval chunks (one topic per chunk)."""
import re


def split_sections(text: str, max_chars: int) -> tuple[str, list[str]]:
    """One chunk per markdown section ("## Ada-7 ..." and its paragraphs).

    Sections, not fixed-size windows: a window packs unrelated topics into one
    chunk, whose vector then matches neither topic well (see NOTLAR.md).
    Each chunk is prefixed with "<document title> - <section title>" so it is
    self-contained. A section longer than max_chars is split at paragraph
    boundaries with the title repeated. The "Kaynak: <url>" line is returned
    as metadata instead of being embedded.

    Returns (source_url, chunks).
    """
    url, doc_title = "", ""
    sections: list[tuple[str, list[str]]] = []  # (section title, paragraphs)
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        # A heading is a single line; text directly below it (no blank line, common
        # in fetched web pages) belongs to the section body, not to its title.
        heading, _, rest = para.partition("\n") if para.startswith("#") else ("", "", "")
        if para.startswith("Kaynak:") or para.startswith("Source:"):
            url = para.split(":", 1)[1].strip()
        elif para.startswith("# "):
            doc_title = heading[2:].strip()
            if rest.strip():
                sections.append(("", [rest.strip()]))
        elif para.startswith("#"):
            sections.append((heading.lstrip("# ").strip(), [rest.strip()] if rest.strip() else []))
        elif sections:
            sections[-1][1].append(para)
        else:  # text before the first section
            sections.append(("", [para]))

    chunks = []
    for title, paragraphs in sections:
        header = " - ".join(t for t in (doc_title, title) if t)
        current = ""
        for para in (piece for p in paragraphs for piece in _fit(p, max_chars)):
            if current and len(current) + len(para) > max_chars:
                chunks.append(f"{header}\n{current}".strip())
                current = ""
            current += para + "\n"
        if current:
            chunks.append(f"{header}\n{current}".strip())
    return url, chunks


def _fit(paragraph: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph at line ends (list items), then at sentence
    ends. Bounds the context the LLM must read: prefill time grows linearly
    with chunk length (~5 ms per token on the GTX 1650)."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, current = [], ""
    for part in (s for line in paragraph.splitlines()
                 for s in (re.split(r"(?<=[.!?])\s+", line) if len(line) > max_chars else [line])):
        if current and len(current) + len(part) > max_chars:
            pieces.append(current.strip())
            current = ""
        current += part + "\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces
