"""Topic names for follow-up questions: suggested from documents, confirmed by a person.

Topics (robot, product and company names) let "Kaç saat çalışır?" be searched
as "ARAT: Kaç saat çalışır?" (app/agent/topics.py). A name extractor was
measured on the knowledge base: it found most names but ~40% of its output was
noise ("Durum", "QR", "Teknik Çizimler"), and a noisy topic silently rewrites
later questions. So names are suggested, and only names a person confirms in
the admin panel are used.

Active topics = config.yaml followup.topics + the confirmed names stored in
data/topics.json.
"""
import json
import re
from collections import Counter
from pathlib import Path

TOPICS_FILE = Path("data/topics.json")

_UP = "A-ZÇĞİÖŞÜ"
_LOW = "a-zçğıöşü"
# A name word: all caps (AKINSOFT, UV-C, AKINCI-5), a word with a number (Ada-7) or a model code (V3).
_NAME_WORD = rf"(?:[{_UP}][{_UP}0-9]+(?:-[{_UP}0-9]+)*|[{_UP}][{_LOW}]+-\d+|V\d+)"
_CAP_WORD = rf"[{_UP}][{_LOW}]+"
# Up to three words ending in a name word ("Servis Robotu V3", "WOLVOX MRP"),
# or two capitalised words ("Mini Ada").
_NAME = re.compile(rf"(?<![\w-])(?:(?:{_CAP_WORD}|{_NAME_WORD})\s){{0,2}}{_NAME_WORD}(?![\w-])"
                   rf"|(?<![\w-]){_CAP_WORD}\s{_CAP_WORD}(?![\w-])")


def _fold(name: str) -> str:
    return re.sub(r"[\s-]+", "", name.lower().replace("ı", "i").replace("i̇", "i"))


def suggest_topics(text: str, known: list[str], limit: int = 8, min_count: int = 2) -> list[str]:
    """Candidate names that occur at least min_count times in the text and are
    not already known (also not a part of a known name, e.g. "Robotu V3")."""
    known_folded = [_fold(k) for k in known]
    body = re.sub(r"^(Kaynak|Source):.*$", "", text, flags=re.MULTILINE)
    counts = Counter(m.group(0) for m in _NAME.finditer(body))
    out = []
    for name, n in counts.most_common():
        folded = _fold(name)
        if n < min_count or len(folded) < 3 or any(folded in k or k in folded for k in known_folded):
            continue
        out.append(name)
        known_folded.append(folded)
        if len(out) == limit:
            break
    return out


class TopicStore:
    """Confirmed topic names, persisted as JSON."""

    def __init__(self, path: Path = TOPICS_FILE):
        self.path = path

    def load(self) -> list[str]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return []

    def save(self, names: list[str]) -> None:
        unique = list(dict.fromkeys(n.strip() for n in names if n.strip()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(unique, ensure_ascii=False, indent=1), encoding="utf-8")


def active_topics(cfg: dict, store: TopicStore | None = None) -> list[str]:
    configured = cfg.get("followup", {}).get("topics", [])
    return list(dict.fromkeys([*configured, *(store or TopicStore()).load()]))
