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

# Directly-scraped categories: list members and take a balanced sample.
DIRECT_CATEGORIES = ["Bosses", "Talismans", "Sorceries", "Incantations"]
PER_DIRECT_CATEGORY = 30

# Weapons need individual weapon pages (with per-weapon scaling +
# requirement stats), not the type-overview "list" pages -- those live in
# type subcategories nested under Melee Armaments / Ranged Weapons /
# Catalysts (e.g. Category:Great Hammers -> Brick Hammer, Giant-Crusher).
# Discovered dynamically so new weapon types are picked up automatically.
WEAPON_PARENT_CATEGORIES = ["Melee Armaments", "Ranged Weapons", "Catalysts"]
# not real weapons for build purposes (arrows/bolts)
WEAPON_TYPE_EXCLUDE = {"Ammunition"}
PER_WEAPON_TYPE = 8

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RATE_LIMIT_SECONDS = 0.8

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


def list_category_members(category: str, limit_per_call: int = 100, cmtype: str = "page") -> list[str]:
    """List member titles of a category. cmtype='page' returns articles
    only (no nested subcategories or files); cmtype='subcat' returns
    subcategory titles."""
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": cmtype,
            "cmlimit": limit_per_call,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        time.sleep(RATE_LIMIT_SECONDS)

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            title = m["title"]
            # when listing subcategories, titles legitimately start with
            # "Category:" -- only apply the skip-prefix filter to page
            # listings (where a stray Category:/Template:/File: is noise)
            if cmtype == "subcat" or not title.startswith(SKIP_TITLE_PREFIXES):
                titles.append(title)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles


def discover_weapon_types() -> list[str]:
    """Weapon type subcategories (Great Hammers, Katanas, ...) nested under
    the weapon parent categories."""
    types: list[str] = []
    for parent in WEAPON_PARENT_CATEGORIES:
        for sub in list_category_members(parent, cmtype="subcat"):
            name = sub.replace("Category:", "")
            if name not in WEAPON_TYPE_EXCLUDE and name not in types:
                types.append(name)
    return types


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

    title_to_category: dict[str, str] = {}
    ordered_titles: list[str] = []

    def add(title: str, category: str):
        if title not in title_to_category:
            title_to_category[title] = category
            ordered_titles.append(title)

    # 1) Individual weapons from each type subcategory (the Phase 7 win --
    #    per-weapon scaling/requirement stats, not just type-overview lists).
    print("Discovering weapon types...")
    weapon_types = discover_weapon_types()
    print(f"  found {len(weapon_types)} weapon types")
    for wtype in weapon_types:
        members = list_category_members(wtype)
        # skip the type's own overview page (title == type name)
        weapons = [m for m in members if m != wtype][:PER_WEAPON_TYPE]
        for w in weapons:
            add(w, "Weapons")
    print(f"  {sum(1 for c in title_to_category.values() if c == 'Weapons')} individual weapons")

    # 2) Balanced sample from the direct categories.
    print("Discovering pages from direct categories...")
    for category in DIRECT_CATEGORIES:
        members = list_category_members(category)[:PER_DIRECT_CATEGORY]
        for m in members:
            add(m, category)
        print(f"  {category}: up to {PER_DIRECT_CATEGORY}")

    print(f"\nFetching {len(ordered_titles)} pages...\n")
    manifest = []
    for title in tqdm(ordered_titles, desc="Fetching articles"):
        html = fetch_page_html(title)
        time.sleep(RATE_LIMIT_SECONDS)
        if html is None:
            continue
        slug = slugify(title)
        out_path = OUT_DIR / f"{slug}.html"
        out_path.write_text(html, encoding="utf-8")
        manifest.append({"title": title, "category": title_to_category[title], "slug": slug,
                          "url": f"https://eldenring.fandom.com/wiki/{title.replace(' ', '_')}"})

    manifest_path = OUT_DIR.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved {len(manifest)} pages to {OUT_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
