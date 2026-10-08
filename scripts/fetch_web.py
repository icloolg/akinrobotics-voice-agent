"""Crawl the company websites into markdown documents (config.yaml web_sources).

    python -m scripts.fetch_web        # writes knowledge_web/*.md
    python -m scripts.ingest           # index (only if knowledge_web is in rag.extra_dirs)

Pages: each site's llms.txt (a list of key pages published for AI systems)
minus forms, social media, logins and English copies; the llms.txt itself is
kept, and llms-full.txt is stored as raw text. A paragraph shared by several
pages (slogans, common blocks) is kept only on the first page. One request
per second; robots.txt is honoured. Cleaning rules: app/knowledge/web.py.
"""
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.config import load_config
from app.knowledge.web import (USER_AGENT, clean, extract, get, normalize, page_urls, robots_for,
                               slug, strip_links, to_markdown)


def main() -> None:
    cfg = load_config()["web_sources"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()

    seen_paragraphs: set[str] = set()  # across pages: shared blocks are kept once
    stats = {"pages": 0, "kept": 0, "dropped_shared": 0, "skipped_robots": 0}
    for site in cfg["sites"]:
        robots = robots_for(site)
        llms = get(f"{site}/llms.txt")
        try:
            full = get(f"{site}/llms-full.txt")
            # The first "# " line is the title; deeper headings become sections.
            full = strip_links(re.sub(r"^#\s", "## ", full, flags=re.M).replace("## ", "# ", 1))
            source = f"\n\nKaynak: {site}/llms-full.txt (indirildi: {date.today():%d.%m.%Y})\n"
            (out / f"{slug(site)}_llms-full.md").write_text(full.replace("\n", source, 1), encoding="utf-8")
            print(f"  llms-full.txt: {len(full) // 1000} KB  {site}")
        except requests.RequestException:
            pass
        (out / f"{slug(site)}_llms.md").write_text(
            f"# {urlparse(site).netloc} llms.txt\n\nKaynak: {site}/llms.txt (indirildi: {date.today():%d.%m.%Y})\n\n"
            + strip_links(llms), encoding="utf-8")
        for title, url in page_urls(llms, site, cfg["exclude"]):
            if not robots.can_fetch(USER_AGENT, url):
                print(f"  robots.txt izin vermiyor: {url}")
                stats["skipped_robots"] += 1
                continue
            time.sleep(cfg.get("delay_s", 1.0))
            try:
                html = get(url)
            except requests.RequestException as e:
                print(f"  indirilemedi: {url} ({e})")
                continue
            paragraphs = []
            for p in clean(extract(html), cfg.get("min_line_chars", 25)):
                key = normalize(p)
                if key in seen_paragraphs and not p.startswith("#"):
                    stats["dropped_shared"] += 1
                    continue
                seen_paragraphs.add(key)
                paragraphs.append(p)
            if not paragraphs:
                continue
            (out / f"{slug(url)}.md").write_text(to_markdown(title, url, paragraphs), encoding="utf-8")
            stats["pages"] += 1
            stats["kept"] += len(paragraphs)
            print(f"  {len(paragraphs):3d} paragraf  {url}")
    print(f"\n{stats['pages']} sayfa, {stats['kept']} paragraf; başka sayfada tekrar eden "
          f"{stats['dropped_shared']} paragraf ayıklandı; robots.txt nedeniyle atlanan {stats['skipped_robots']}.")


if __name__ == "__main__":
    main()
