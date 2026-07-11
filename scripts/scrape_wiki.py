"""Corpus collection for the Elden Ring wiki via the MediaWiki API.

Why the API instead of scraping rendered pages: Fandom's robots.txt
explicitly allowlists `/api.php?action=` for all crawlers -- it's the
officially sanctioned programmatic access point, not a workaround. It also
returns clean article-body content (no nav/ads/sidebar to strip), which
makes Phase 2 (cleaning) much simpler.

(For the record: we initially tried scraping rendered /wiki/ pages
directly. Cloudflare's bot-management fingerprinted and 403'd Python's
`requests` client specifically -- identical headers worked fine via curl,
which points to TLS/client fingerprinting rather than a missing header.
We did not chase that by spoofing fingerprints or swapping HTTP clients to
get through it; the API path sidesteps the question entirely since it's
the documented, allowlisted integration point.)

Process: for each category in SEED_CATEGORIES, list its member pages via
action=query&list=categorymembers, then fetch each page's article-body
HTML via action=parse&prop=text. Capped at MAX_PAGES total.

Usage: python scripts/scrape_wiki.py
"""
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

API_URL = "https://eldenring.fandom.com/api.php"
SEED_CATEGORIES = ["Weapons", "Bosses", "Talismans", "Sorceries", "Incantations"]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RATE_LIMIT_SECONDS = 1.0
MAX_PAGES = 150

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Category/list/meta pages that show up as "members" but aren't articles we
# want to answer questions from.
SKIP_TITLE_PREFIXES = ("Category:", "Template:", "File:", "User:", "Help:")


def api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def list_category_members(category: str, limit_per_call: int = 100) -> list[str]:
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit_per_call,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        time.sleep(RATE_LIMIT_SECONDS)

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            title = m["title"]
            if not title.startswith(SKIP_TITLE_PREFIXES):
                titles.append(title)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles


def fetch_page_html(title: str) -> str | None:
    try:
        data = api_get({"action": "parse", "page": title, "prop": "text"})
    except requests.RequestException as e:
        print(f"  failed: {title} ({e})")
        return None
    if "error" in data:
        print(f"  api error for {title}: {data['error'].get('info')}")
        return None
    return data["parse"]["text"]["*"]


WINDOWS_INVALID_CHARS = '<>:"/\\|?*'


def slugify(title: str) -> str:
    slug = title.replace(" ", "_")
    for ch in WINDOWS_INVALID_CHARS:
        slug = slug.replace(ch, "-")
    return slug


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering pages from seed categories...")
    per_category: dict[str, list[str]] = {}
    seen_titles: set[str] = set()
    for category in SEED_CATEGORIES:
        titles = [t for t in list_category_members(category) if t not in seen_titles]
        seen_titles.update(titles)
        per_category[category] = titles
        print(f"  {category}: {len(titles)} pages")

    # round-robin across categories so no single category (e.g. Bosses, the
    # largest) exhausts the MAX_PAGES budget before the others are sampled
    title_to_category: dict[str, str] = {}
    titles_to_fetch: list[str] = []
    queues = {c: list(ts) for c, ts in per_category.items()}
    while len(titles_to_fetch) < MAX_PAGES and any(queues.values()):
        for category in SEED_CATEGORIES:
            if not queues[category]:
                continue
            title = queues[category].pop(0)
            titles_to_fetch.append(title)
            title_to_category[title] = category
            if len(titles_to_fetch) >= MAX_PAGES:
                break

    all_titles = title_to_category
    print(f"\nFetching {len(titles_to_fetch)} pages (capped at {MAX_PAGES}, balanced across categories)...\n")

    manifest = []
    for title in tqdm(titles_to_fetch, desc="Fetching articles"):
        html = fetch_page_html(title)
        time.sleep(RATE_LIMIT_SECONDS)
        if html is None:
            continue
        slug = slugify(title)
        out_path = OUT_DIR / f"{slug}.html"
        out_path.write_text(html, encoding="utf-8")
        manifest.append({"title": title, "category": all_titles[title], "slug": slug,
                          "url": f"https://eldenring.fandom.com/wiki/{title.replace(' ', '_')}"})

    manifest_path = OUT_DIR.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved {len(manifest)} pages to {OUT_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
