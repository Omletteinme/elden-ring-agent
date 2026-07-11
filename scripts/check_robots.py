"""Fetch and print robots.txt for a target domain before scraping it.

Usage: python scripts/check_robots.py <base_url>
"""
import sys
import urllib.robotparser as robotparser


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "https://eldenring.wiki.fextralife.com"
    robots_url = base.rstrip("/") + "/robots.txt"
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as e:
        print(f"Could not fetch {robots_url}: {e}")
        sys.exit(1)

    test_paths = ["/", "/Weapons", "/Bosses", "/Builds"]
    for path in test_paths:
        allowed = rp.can_fetch("*", base + path)
        print(f"{'ALLOWED' if allowed else 'DISALLOWED':<12} {path}")

    delay = rp.crawl_delay("*")
    print(f"\nCrawl-delay: {delay if delay else 'none specified (default to 1-2s between requests)'}")


if __name__ == "__main__":
    main()
