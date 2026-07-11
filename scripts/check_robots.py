"""Fetch and print robots.txt for a target domain before scraping it.

Note: Python's stdlib urllib.robotparser doesn't reliably parse Fandom's
robots.txt into usable entries (0 entries either way we tried it -- see
scrape_wiki.fetch_robots_txt() for details), and separately, its own
internal fetch uses a bare urllib User-Agent that Cloudflare 403s, which
robotparser then (mis)treats as "disallow everything". This script fetches
the raw file with a normal browser header so you can read the real rules
yourself rather than trusting an automated parse.

Usage: python scripts/check_robots.py <base_url>
"""
import sys

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "https://eldenring.fandom.com"
    robots_url = base.rstrip("/") + "/robots.txt"
    resp = requests.get(robots_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    print(resp.text)


if __name__ == "__main__":
    main()
