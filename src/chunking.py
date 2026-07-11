"""Clean and chunk scraped Elden Ring wiki article HTML.

Each raw page (article-body HTML from the MediaWiki API) gets split into:
  - one "infobox" chunk: the stat block (Type, FP Cost, attributes required,
    etc.) rendered as readable "Label: Value" lines -- this is the highest
    value content for exact-answer questions ("what's the FP cost of X").
  - one or more "section" chunks: the prose body, split at headings
    (Description, Acquisition, Location, ...) and further split by
    paragraph/list boundaries if a section runs long.

Noise removed: edit-section "[edit]" links, the mobile-hidden duplicate
flavor-text block (language-flag clutter), the trailing "v-d-e" navbox
listing every page in the category (not page-specific content), and empty
placeholder paragraphs.
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.json"

MAX_CHUNK_CHARS = 900
NOISE_CLASSES = {"mobile-hidden", "hidden", "mw-empty-elt", "mw-editsection"}


@dataclass
class Chunk:
    id: str
    title: str
    category: str
    url: str
    section: str
    text: str


NAVBOX_LINK_THRESHOLD = 20


def _strip_noise(root: Tag) -> None:
    # remove MediaWiki's HTML comments (e.g. the "NewPP limit report" /
    # cache-timing block) -- bs4's get_text() includes Comment text by
    # default since Comment subclasses NavigableString, so these leak into
    # chunks unless stripped explicitly.
    for comment in root.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # collect matches before decomposing -- decompose() nulls out a
    # subtree's internal state, so mutating while iterating a find_all()
    # snapshot that still references now-dead descendants crashes
    to_remove = [el for el in root.find_all(True) if set(el.get("class") or []) & NOISE_CLASSES]
    for el in to_remove:
        # a match may already be gone if an earlier match was its ancestor
        if el.parent is not None:
            el.decompose()


def _remove_trailing_navbox(root: Tag) -> None:
    """Remove the 'v-d-e <Category> in Elden Ring [full page list]' navbox
    (e.g. every spell/weapon/boss name linked at the bottom of the page).

    It shares its table class ("article-table mw-collapsible") with
    legitimate content tables like patch-notes tables, so class alone
    can't distinguish it. Navboxes are link-dense (50-150+ <a> tags across
    the whole category); real content tables have a handful at most -- we
    use that as the signal instead, checked only at the top level so we
    don't accidentally gut a legitimate section for being link-heavy.
    """
    for el in root.find_all(["div", "table"], recursive=False):
        text = el.get_text(" ", strip=True)
        link_count = len(el.find_all("a"))
        if text.startswith(("v · d · e", "v·d·e", "v d e")) or link_count > NAVBOX_LINK_THRESHOLD:
            el.decompose()


def _table_to_text(table: Tag) -> str:
    rows = table.find_all("tr")
    if not rows:
        return table.get_text(" ", strip=True)
    header_cells = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
    lines = []
    for row in rows[1:]:
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        if not cells:
            continue
        if len(cells) == len(header_cells):
            pieces = [f"{h}: {v}" for h, v in zip(header_cells, cells) if v]
            lines.append(", ".join(pieces))
        else:
            lines.append(" | ".join(cells))
    return "; ".join(lines)


def extract_infobox_text(soup: BeautifulSoup, title: str) -> str | None:
    aside = soup.find("aside", class_="portable-infobox")
    if aside is None:
        return None

    labels: dict[str, str] = {}
    values: dict[str, str] = {}
    merged_lines: list[str] = []

    for el in aside.select("[data-source]"):
        ds = el.get("data-source")
        if ds in ("title", "image", "japanese"):
            continue
        classes = el.get("class") or []
        is_label = any("label" in c for c in classes)
        is_value = any("value" in c for c in classes)
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if is_label:
            labels[ds] = text
        elif is_value:
            values[ds] = text
        else:
            nested_label = el.select_one(".pi-data-label")
            nested_value = el.select_one(".pi-data-value")
            if nested_label and nested_value:
                merged_lines.append(f"{nested_label.get_text(' ', strip=True)}: {nested_value.get_text(' ', strip=True)}")
            else:
                merged_lines.append(text)

    for ds in labels.keys() | values.keys():
        label = labels.get(ds, ds)
        value = values.get(ds, "")
        if value:
            merged_lines.append(f"{label}: {value}")

    if not merged_lines:
        return None
    return f"{title} — " + "; ".join(merged_lines)


def extract_sections(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """Returns a list of (section_name, section_text), in document order,
    with the infobox and noise already removed from `soup`."""
    root = soup.find("div", class_="mw-parser-output") or soup

    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for node in root.children:
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                sections[-1][1].append(text)
            continue
        if not isinstance(node, Tag):
            continue

        if node.name in ("h2", "h3", "h4"):
            headline = node.select_one(".mw-headline")
            name = headline.get_text(" ", strip=True) if headline else node.get_text(" ", strip=True)
            if name:
                sections.append((name, []))
            continue

        if node.name == "table":
            text = _table_to_text(node)
        else:
            text = node.get_text(" ", strip=True)

        # "Show Patch Notes" is a collapsible-table toggle button label,
        # not content -- the table's own text (extracted separately above)
        # already carries the actual patch info.
        if text and text != "Show Patch Notes":
            sections[-1][1].append(text)

    return [(name, " ".join(parts)) for name, parts in sections if " ".join(parts).strip()]


def split_into_chunks(section_text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(section_text) <= max_chars:
        return [section_text]
    # split on sentence-ish boundaries, then pack greedily up to max_chars
    pieces = re.split(r"(?<=[.!?])\s+", section_text)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def process_page(html: str, title: str, category: str, url: str) -> list[Chunk]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("div", class_="mw-parser-output") or soup

    _strip_noise(root)

    chunks: list[Chunk] = []

    infobox_text = extract_infobox_text(soup, title)
    aside = root.find("aside", class_="portable-infobox")
    if aside is not None:
        aside.decompose()
    if infobox_text:
        chunks.append(Chunk(
            id=f"{title}::infobox", title=title, category=category, url=url,
            section="Infobox", text=infobox_text,
        ))

    _remove_trailing_navbox(root)

    for section_name, section_text in extract_sections(soup):
        for i, piece in enumerate(split_into_chunks(section_text)):
            chunks.append(Chunk(
                id=f"{title}::{section_name}::{i}", title=title, category=category,
                url=url, section=section_name, text=piece,
            ))

    return chunks


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    all_chunks: list[dict] = []
    for entry in manifest:
        raw_path = RAW_DIR / f"{entry['slug']}.html"
        if not raw_path.exists():
            print(f"  missing raw file for {entry['title']}, skipping")
            continue
        html = raw_path.read_text(encoding="utf-8")
        chunks = process_page(html, entry["title"], entry["category"], entry["url"])
        all_chunks.extend(asdict(c) for c in chunks)

    out_path = CHUNK_DIR / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks from {len(manifest)} pages to {out_path}")
    avg_len = sum(len(c["text"]) for c in all_chunks) / max(len(all_chunks), 1)
    print(f"Average chunk length: {avg_len:.0f} chars")


if __name__ == "__main__":
    main()
