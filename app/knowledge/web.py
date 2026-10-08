"""Web pages as knowledge documents (main text only, polite crawling).

Used by scripts/fetch_web.py (site crawl) and the admin panel ("add URL").

Cleaning:
  - main text only: trafilatura removes menus, footers and cookie banners;
  - repeated paragraphs are dropped (within a page, and across pages in a crawl);
  - short lines without context are dropped: spec tables lose their labels on
    extraction ("65 KG", "16.12.2019"), and a bare value can mislead the model.

robots.txt is honoured. It is fetched with our own User-Agent: Python's
RobotFileParser.read() uses a default agent that some sites refuse, after
which it reports every URL as disallowed.
"""
import re
from datetime import date
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura

USER_AGENT = "AkinVoice/1.0 (assessment project; respects robots.txt)"
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def strip_links(text: str) -> str:
    """"[Title](https://...)" -> "Title": URLs add noise to embeddings."""
    return LINK.sub(lambda m: m.group(1).strip(), text)


def get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def robots_for(site: str) -> RobotFileParser:
    rp = RobotFileParser()
    try:
        rp.parse(get(f"{site}/robots.txt").splitlines())
    except requests.RequestException:
        rp.parse([])  # no robots.txt: everything is allowed
    return rp


def page_urls(llms_txt: str, site: str, exclude: list[str]) -> list[tuple[str, str]]:
    """(title, url) of the site's own pages listed in its llms.txt."""
    host = urlparse(site).netloc
    seen, out = set(), []
    for title, url in LINK.findall(llms_txt):
        url = url.rstrip("/") if url.count("/") > 3 else url
        path = urlparse(url).path.rstrip("/")
        if (urlparse(url).netloc != host or url in seen or any(x in url for x in exclude)
                or path in ("/en", "") and url != site or path.endswith((".xml", ".txt"))):
            continue
        seen.add(url)
        out.append((title.split("|")[0].strip(), url))
    return out


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def clean(markdown: str, min_chars: int) -> list[str]:
    """Paragraphs of a page without exact repeats and context-free short lines."""
    paragraphs, seen = [], set()
    for p in (p.strip() for p in markdown.split("\n\n")):
        key = normalize(p)
        if not key or key in seen:
            continue
        if not p.startswith("#") and len(p) < min_chars:
            continue
        seen.add(key)
        paragraphs.append(p)
    return paragraphs


def extract(html: str) -> str:
    return trafilatura.extract(html, output_format="markdown", include_tables=True,
                               include_links=False, include_images=False) or ""


def to_markdown(title: str, url: str, paragraphs: list[str]) -> str:
    # Page headings become "##" sections; "# " is reserved for the document title.
    body = [re.sub(r"^#+\s*", "## ", p) if p.startswith("#") else p for p in paragraphs]
    return f"# {title}\n\nKaynak: {url} (indirildi: {date.today():%d.%m.%Y})\n\n" + "\n\n".join(body) + "\n"


def slug(url: str) -> str:
    u = urlparse(url)
    path = u.path.strip("/").replace("/", "_") or "anasayfa"
    return re.sub(r"[^\w.-]+", "-", f"{u.netloc.replace('www.', '')}_{path}")[:120]


def fetch_page(url: str, min_chars: int = 25) -> tuple[str, int]:
    """One page as a markdown document: (markdown, paragraph count).
    Raises ValueError for a non-http URL, PermissionError if robots.txt disallows it."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise ValueError("http(s) adresi olmalı")
    if not robots_for(f"{u.scheme}://{u.netloc}").can_fetch(USER_AGENT, url):
        raise PermissionError("robots.txt bu sayfaya izin vermiyor")
    html = get(url)
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else u.netloc).split("|")[0].strip()
    paragraphs = clean(extract(html), min_chars)
    return to_markdown(title, url, paragraphs), len(paragraphs)
