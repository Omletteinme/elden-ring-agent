"""Structured weapon recommendation by attribute scaling.

Pure retrieval can't answer "best weapon for a strength build" well: the
per-weapon stat chunks are textually near-identical ("Requires: ...;
Scaling: ..."), so semantic/keyword search returns an arbitrary set of
strength-ish weapons, not the *best* ones. Ranking by scaling grade is a
computation, not a similarity match -- so this module parses the weapon
attribute chunks into structured records and ranks them deterministically
(S > A > B > C > D > E). Exposed to the agent as a second tool
(recommend_weapons) alongside search_wiki.

Scaling grades here are base grades (before affinities/upgrades), which is
what the wiki infobox lists -- honest and consistent, if not the full
min-max picture.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.jsonl"

GRADE_ORDER = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
CANONICAL_ATTR = {
    "strength": "Strength", "str": "Strength",
    "dexterity": "Dexterity", "dex": "Dexterity",
    "intelligence": "Intelligence", "int": "Intelligence",
    "faith": "Faith", "fai": "Faith", "fth": "Faith",
    "arcane": "Arcane", "arc": "Arcane",
}


def _parse_scaling(text: str) -> dict[str, str]:
    m = re.search(r"Scaling:\s*(.+)$", text)
    if not m:
        return {}
    scaling = {}
    for part in m.group(1).split(","):
        toks = part.strip().rsplit(" ", 1)
        if len(toks) == 2 and toks[1] in GRADE_ORDER:
            scaling[toks[0].strip()] = toks[1]
    return scaling


def _parse_requirement(text: str, attribute: str) -> str | None:
    m = re.search(r"Requires:\s*(.+?);\s*Scaling", text)
    if not m:
        return None
    for part in m.group(1).split(","):
        part = part.strip()
        if part.startswith(attribute):
            return part[len(attribute):].strip()
    return None


@lru_cache(maxsize=1)
def _load_weapon_stats() -> list[dict]:
    weapons = []
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if c.get("section") != "Attributes":
            continue
        scaling = _parse_scaling(c["text"])
        if scaling:
            weapons.append({"title": c["title"], "url": c["url"], "scaling": scaling, "text": c["text"]})
    return weapons


def recommend_weapons(attribute: str, limit: int = 8) -> list[dict]:
    """Weapons that scale in `attribute`, ranked best scaling first."""
    canonical = CANONICAL_ATTR.get(attribute.strip().lower())
    if canonical is None:
        return []
    matches = [w for w in _load_weapon_stats() if canonical in w["scaling"]]
    matches.sort(key=lambda w: GRADE_ORDER.get(w["scaling"][canonical], 0), reverse=True)
    out = []
    for w in matches[:limit]:
        out.append({
            "title": w["title"],
            "attribute": canonical,
            "scaling_grade": w["scaling"][canonical],
            "requirement": _parse_requirement(w["text"], canonical),
            "url": w["url"],
        })
    return out


if __name__ == "__main__":
    import sys
    attr = sys.argv[1] if len(sys.argv) > 1 else "strength"
    for r in recommend_weapons(attr, limit=10):
        print(f"  {r['scaling_grade']}  {r['title']} (requires {r['attribute']} {r['requirement']})")
