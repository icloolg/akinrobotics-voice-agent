"""PDF to markdown: the text layer of each page, or OCR for image-only pages.

Brochures and manuals are often exported as images (the robot arm catalog and
the Mini Ada manual have no text layer). Such pages are rendered and read with
EasyOCR (Turkish + English). OCR costs ~7-12 s per page on the CPU, so results
are cached in data/ocr_cache/ keyed by the file's content hash.

Output: "# <file>" plus one "## Sayfa N" section per page, so PDFs go through
the same section chunking as markdown documents.
"""
import hashlib
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/ocr_cache")
MIN_TEXT_CHARS = 20  # a page with less text is treated as an image
_ocr_reader = None   # loaded lazily, only when a page needs OCR

Progress = Callable[[str], None]


def pdf_to_markdown(path: Path, progress: Progress | None = None) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < MIN_TEXT_CHARS:
            if progress:
                progress(f"{path.name}: OCR {i + 1}/{len(reader.pages)}")
            text = _ocr_page(path, i)
        if text:
            pages.append(f"## Sayfa {i + 1}\n\n{text}")
    return f"# {path.stem}\n\nKaynak: {path.name}\n\n" + "\n\n".join(pages)


def _ocr_page(path: Path, index: int) -> str:
    key = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    cached = CACHE_DIR / f"{path.stem}-{key}-p{index + 1}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    import pypdfium2 as pdfium
    import numpy as np

    image = pdfium.PdfDocument(path)[index].render(scale=2).to_pil()  # ~144 dpi: small text stays readable
    text = layout_text(_reader().readtext(np.array(image), paragraph=False))
    log.info("OCR %s page %d: %d characters", path.name, index + 1, len(text))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


class _Box:
    def __init__(self, points, text):
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        self.x0, self.x1, self.y0, self.y1 = min(xs), max(xs), min(ys), max(ys)
        self.text = text.strip()

    @property
    def h(self):
        return self.y1 - self.y0

    def overlaps_x(self, other) -> bool:
        overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        return overlap > 0.3 * min(self.x1 - self.x0, other.x1 - other.x0)


def layout_text(results) -> str:
    """OCR boxes -> lines, pairing spec-sheet labels with their values.

    Spec sheets place a small label above a large value ("Ağırlık / Weight"
    over "27 kg"). Plain row order turns a two-column sheet into "Ağırlık,
    Taşıma Kapasitesi, 27 kg, 5 kg", which the LLM paired wrongly. Each box
    therefore takes as its label the nearest clearly smaller box above it in
    the same column ("Ağırlık / Weight: 27 kg"); the smaller bilingual repeat
    directly under a paired value is dropped.
    """
    boxes = sorted((_Box(p, t) for p, t, *_ in results if t.strip()), key=lambda b: (b.y0, b.x0))

    # 1. Join boxes on the same line that touch ("Taşıma Mekanizması" + "Handling Mechanism").
    merged: list[_Box] = []
    for b in boxes:
        last = next((m for m in reversed(merged) if abs(m.y0 - b.y0) < 6 and 0 <= b.x0 - m.x1 < 20), None)
        if last:
            last.text += " " + b.text
            last.x1, last.y1 = b.x1, max(last.y1, b.y1)
        else:
            merged.append(b)

    # 2. Pair each value with the small label right above it.
    label_of: dict[int, _Box] = {}
    used: set[int] = set()
    for i, b in enumerate(merged):
        above = [m for m in merged[:i] if m.y1 <= b.y0 + 5 and b.y0 - m.y1 <= b.h and m.overlaps_x(b)]
        if above:
            nearest = max(above, key=lambda m: m.y1)
            if nearest.h < 0.7 * b.h and id(nearest) not in used:
                label_of[id(b)] = nearest
                used.add(id(nearest))

    # 3. Write lines; skip used labels and the small repeat under a paired value.
    lines, values = [], [b for b in merged if id(b) in label_of]
    for b in merged:
        if id(b) in used:
            continue
        # "Directly under" = within half a line of the value's bottom; a wider
        # window also dropped real values below large titles.
        if any(v is not b and v.overlaps_x(b) and abs(b.y0 - v.y1) <= b.h / 2 and b.h < v.h for v in values):
            continue
        lines.append(f"{label_of[id(b)].text}: {b.text}" if id(b) in label_of else b.text)
    return "\n".join(lines)


def _reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["tr", "en"], gpu=False, verbose=False)
    return _ocr_reader
