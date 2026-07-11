"""Polite, rate-limited scraper for the Elden Ring wiki.

Starts from a small set of seed/category pages, discovers same-wiki links
that match the content-page pattern, and downloads each page once — capped
at MAX_PAGES so this stays a focused subset (weapons/bosses/builds), not
the entire wiki.

Respects robots.txt (aborts on disallowed paths) and rate-limits requests.

Usage: python scripts/scrape_wiki.py
"""
import time
import urllib.robotparser as robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://eldenring.wiki.fextralife.com"
SEED_PATHS = ["/Weapons", "/Bosses", "/Builds", "/Talismans", "/Spells"]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
USER_AGENT = "elden-ring-agent-portfolio-project/0.1 (personal research/eval project; contact via GitHub Omletteinme)"
RATE_LIMIT_SECONDS = 1.5
MAX_PAGES = 150

HEADERS = {"User-Agent": USER_AGENT}


def load_robots():
    rp = robotparser.RobotFileParser()
    rp.set_url(BASE_URL + "/robots.txt")
    rp.read()
    return rp


def is_content_link(href: str) -> bool:
    """Heuristic: same-wiki article links, not files/edit/history/talk pages."""
    if not href or href.startswith("#"):
        return False
    parsed = urlparse(urljoin(BASE_URL, href))
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    path = parsed.path
    bad_markers = ("/File:", "?", "&", "/Special:", "/Talk:", "action=edit", "action=history")
    if any(m in href for m in bad_markers):
        return False
    return path.count("/") >= 1 and len(path) > 1


def discover_links(html: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.select("#wiki-content-block a[href], .wiki_text a[href]"):
        href = a.get("href")
        if is_content_link(href):
            links.add(urljoin(BASE_URL, href).split("#")[0])
    return links


def fetch(url: str, rp: robotparser.RobotFileParser) -> str | None:
    if not rp.can_fetch(USER_AGENT, url):
        print(f"  skip (robots.txt disallows): {url}")
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  failed: {url} ({e})")
        return None


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.replace("/", "_") or "index"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Checking robots.txt...")
    rp = load_robots()

    to_visit = {urljoin(BASE_URL, p) for p in SEED_PATHS}
    visited: set[str] = set()
    queue = list(to_visit)

    pbar = tqdm(total=MAX_PAGES, desc="Scraping pages")
    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = fetch(url, rp)
        time.sleep(RATE_LIMIT_SECONDS)
        if html is None:
            continue

        out_path = OUT_DIR / f"{slugify(url)}.html"
        out_path.write_text(html, encoding="utf-8")
        pbar.update(1)

        # only expand the frontier from seed/category pages to keep this a
        # focused subset rather than crawling the whole wiki
        if url in to_visit:
            new_links = discover_links(html)
            for link in new_links:
                if link not in visited and link not in queue:
                    queue.append(link)

    pbar.close()
    print(f"\nSaved {len(visited)} pages to {OUT_DIR}")


if __name__ == "__main__":
    main()
