#!/usr/bin/env python
"""Build RPG corpus from Open5e API for RAG evaluation.

Searches the Open5e API for monsters, spells, and other content, renders each
entry to a markdown document with proper stat block formatting, and saves to
a corpus directory with metadata.json.

Usage:
    uv run python build_open5e.py monsters --source wotc-srd --corpus srd
    uv run python build_open5e.py spells --source wotc-srd --corpus srd_spells
    uv run python build_open5e.py monsters --source tob,cc --corpus kobold

Output:
    <data-dir>/<corpus>/
        metadata.json   - Corpus metadata with file counts and licensing
        docs/
            <slug>.md   - Individual markdown documents

API Notes:
    - Base URL: https://api.open5e.com/v1/
    - Pagination: ?limit=100&page=N
    - Filtering: ?document__slug=wotc-srd
    - No authentication required
"""

import argparse
import contextlib
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OPEN5E_BASE_URL = "https://api.open5e.com/v1"

# Rate limiting - Open5e is a free community project, be very respectful
# These are volunteers running this service, not a commercial API
BASE_DELAY_SECONDS = 3.0  # Conservative delay between requests
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.5
MAX_BACKOFF_SECONDS = 300  # 5 minute max backoff


def request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """Make HTTP request with exponential backoff on errors."""
    delay = BASE_DELAY_SECONDS
    last_exception: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            sleep_time = delay
            logger.info(f"Retrying in {sleep_time:.1f}s (attempt {attempt + 1})")
            time.sleep(sleep_time)
            delay = min(delay * BACKOFF_FACTOR, MAX_BACKOFF_SECONDS)
        else:
            time.sleep(BASE_DELAY_SECONDS)

        try:
            response = client.get(url, params=params)

            if response.status_code == 429:
                # Respect Retry-After header if present
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    with contextlib.suppress(ValueError):
                        delay = max(float(retry_after), delay)
                last_exception = httpx.HTTPStatusError(
                    "Rate limited",
                    request=response.request,
                    response=response,
                )
                continue

            if response.status_code >= 500:
                last_exception = httpx.HTTPStatusError(
                    f"Server error ({response.status_code})",
                    request=response.request,
                    response=response,
                )
                continue

            response.raise_for_status()
            return response

        except httpx.TimeoutException as e:
            last_exception = e
            continue
        except httpx.RequestError as e:
            last_exception = e
            continue

    if last_exception:
        raise last_exception
    raise httpx.HTTPError("All retries exhausted")


def _ability_with_mod(score: int) -> str:
    """Format ability score with modifier."""
    mod = (score - 10) // 2
    sign = "+" if mod >= 0 else ""
    return f"{score} ({sign}{mod})"


def _render_monster_header(monster: dict[str, Any]) -> list[str]:
    """Render monster title and type line."""
    lines = [f"# {monster['name']}", ""]
    doc_title = monster.get("document__title", "Unknown")
    doc_slug = monster.get("document__slug", "unknown")
    lines.extend([f"**Source:** {doc_title} ({doc_slug})", ""])

    size = monster.get("size", "Medium")
    creature_type = monster.get("type", "creature")
    subtype = monster.get("subtype", "")
    alignment = monster.get("alignment", "unaligned")
    type_line = f"*{size} {creature_type}"
    if subtype:
        type_line += f" ({subtype})"
    type_line += f", {alignment}*"
    lines.extend([type_line, ""])
    return lines


def _render_monster_combat_stats(monster: dict[str, Any]) -> list[str]:
    """Render AC, HP, Speed, and ability scores."""
    lines = ["---", ""]

    ac = monster.get("armor_class", 10)
    ac_desc = monster.get("armor_desc", "")
    if ac_desc:
        lines.append(f"**Armor Class** {ac} ({ac_desc})")
    else:
        lines.append(f"**Armor Class** {ac}")

    hp = monster.get("hit_points", 1)
    hit_dice = monster.get("hit_dice", "")
    if hit_dice:
        lines.append(f"**Hit Points** {hp} ({hit_dice})")
    else:
        lines.append(f"**Hit Points** {hp}")

    speed_parts = []
    speed_data = monster.get("speed", {})
    if isinstance(speed_data, dict):
        if speed_data.get("walk"):
            speed_parts.append(f"{speed_data['walk']} ft.")
        for mode in ["fly", "swim", "climb", "burrow"]:
            if speed_data.get(mode):
                speed_parts.append(f"{mode} {speed_data[mode]} ft.")
        if speed_data.get("hover"):
            speed_parts.append("(hover)")
    lines.extend(
        [f"**Speed** {', '.join(speed_parts) if speed_parts else '0 ft.'}", ""]
    )

    lines.extend(
        [
            "---",
            "",
            "| STR | DEX | CON | INT | WIS | CHA |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    abilities = [
        monster.get(a, 10)
        for a in [
            "strength",
            "dexterity",
            "constitution",
            "intelligence",
            "wisdom",
            "charisma",
        ]
    ]
    lines.extend([f"| {' | '.join(_ability_with_mod(a) for a in abilities)} |", ""])
    return lines


def _render_monster_defenses(monster: dict[str, Any]) -> list[str]:
    """Render saving throws, skills, and defensive properties."""
    lines = ["---", ""]

    saves = []
    for ability in [
        "strength",
        "dexterity",
        "constitution",
        "intelligence",
        "wisdom",
        "charisma",
    ]:
        if monster.get(f"{ability}_save"):
            saves.append(f"{ability[:3].upper()} +{monster[f'{ability}_save']}")
    if saves:
        lines.append(f"**Saving Throws** {', '.join(saves)}")

    skills = monster.get("skills", {})
    if skills:
        lines.append(
            f"**Skills** {', '.join(f'{k.title()} +{v}' for k, v in skills.items())}"
        )

    for field, label in [
        ("damage_vulnerabilities", "Damage Vulnerabilities"),
        ("damage_resistances", "Damage Resistances"),
        ("damage_immunities", "Damage Immunities"),
        ("condition_immunities", "Condition Immunities"),
        ("senses", "Senses"),
        ("languages", "Languages"),
    ]:
        if monster.get(field):
            lines.append(f"**{label}** {monster[field]}")

    lines.extend([f"**Challenge** {monster.get('challenge_rating', '0')}", ""])
    return lines


def _render_action_block(
    actions: list[dict[str, Any]],
    heading: str,
    default_name: str,
    preamble: str = "",
) -> list[str]:
    """Render a block of actions with a heading."""
    if not actions:
        return []
    lines = [f"## {heading}", ""]
    if preamble:
        lines.extend([preamble, ""])
    for action in actions:
        name = action.get("name", default_name)
        desc = action.get("desc", "")
        lines.extend([f"***{name}.*** {desc}", ""])
    return lines


def _render_lair_actions(lair_actions: list[Any]) -> list[str]:
    """Render lair actions which can be strings or dicts."""
    if not lair_actions:
        return []
    lines = ["## Lair Actions", ""]
    for action in lair_actions:
        if isinstance(action, str):
            lines.append(f"- {action}")
        elif isinstance(action, dict):
            name = action.get("name", "")
            desc = action.get("desc", "")
            lines.append(f"***{name}.*** {desc}" if name else f"- {desc}")
    lines.append("")
    return lines


def render_monster_to_markdown(monster: dict[str, Any]) -> str:
    """Render a monster JSON entry to markdown with stat block formatting."""
    lines: list[str] = []
    lines.extend(_render_monster_header(monster))
    lines.extend(_render_monster_combat_stats(monster))
    lines.extend(_render_monster_defenses(monster))

    # Special abilities
    special = monster.get("special_abilities", [])
    if special:
        lines.extend(["---", ""])
        for ability in special:
            name = ability.get("name", "Ability")
            desc = ability.get("desc", "")
            lines.extend([f"***{name}.*** {desc}", ""])

    lines.extend(_render_action_block(monster.get("actions", []), "Actions", "Action"))
    lines.extend(
        _render_action_block(
            monster.get("bonus_actions", []), "Bonus Actions", "Bonus Action"
        )
    )
    lines.extend(
        _render_action_block(monster.get("reactions", []), "Reactions", "Reaction")
    )
    lines.extend(
        _render_action_block(
            monster.get("legendary_actions", []),
            "Legendary Actions",
            "Legendary Action",
            monster.get("legendary_desc", ""),
        )
    )
    lines.extend(
        _render_action_block(
            monster.get("mythic_actions", []), "Mythic Actions", "Mythic Action"
        )
    )
    lines.extend(_render_lair_actions(monster.get("lair_actions", [])))

    return "\n".join(lines)


def render_spell_to_markdown(spell: dict[str, Any]) -> str:
    """Render a spell JSON entry to markdown."""
    lines = []

    # Title and source
    lines.append(f"# {spell['name']}")
    lines.append("")
    doc_title = spell.get("document__title", "Unknown")
    doc_slug = spell.get("document__slug", "unknown")
    lines.append(f"**Source:** {doc_title} ({doc_slug})")
    lines.append("")

    # Spell level and school
    level = spell.get("level", "Cantrip")
    school = spell.get("school", "")
    ritual = spell.get("ritual", "no")
    level_line = f"*{level} {school}"
    if ritual == "yes":
        level_line += " (ritual)"
    level_line += "*"
    lines.append(level_line)
    lines.append("")

    # Spell properties
    lines.append("---")
    lines.append("")
    lines.append(f"**Casting Time:** {spell.get('casting_time', '1 action')}")
    lines.append(f"**Range:** {spell.get('range', 'Self')}")

    # Components
    components = spell.get("components", "")
    material = spell.get("material", "")
    if material:
        lines.append(f"**Components:** {components} ({material})")
    else:
        lines.append(f"**Components:** {components}")

    lines.append(f"**Duration:** {spell.get('duration', 'Instantaneous')}")
    lines.append("")

    # Classes
    dnd_class = spell.get("dnd_class", "")
    if dnd_class:
        lines.append(f"**Classes:** {dnd_class}")
        lines.append("")

    # Description
    lines.append("---")
    lines.append("")
    desc = spell.get("desc", "")
    lines.append(desc)
    lines.append("")

    # At higher levels
    higher_level = spell.get("higher_level", "")
    if higher_level:
        lines.append("**At Higher Levels.** " + higher_level)
        lines.append("")

    return "\n".join(lines)


def fetch_content(
    client: httpx.Client,
    content_type: str,
    sources: list[str] | None,
    max_docs: int | None,
) -> list[dict[str, Any]]:
    """Fetch all content of a given type, optionally filtered by source."""
    results: list[dict[str, Any]] = []
    page = 1
    limit = 100

    endpoint = f"{OPEN5E_BASE_URL}/{content_type}/"

    with tqdm(desc=f"Fetching {content_type}", unit="items") as pbar:
        while True:
            params: dict[str, Any] = {"limit": limit, "page": page}

            try:
                response = request_with_retry(client, endpoint, params=params)
                data = response.json()
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                break

            items = data.get("results", [])
            if not items:
                break

            for item in items:
                # Filter by source if specified
                if sources:
                    item_source = item.get("document__slug", "")
                    if item_source not in sources:
                        continue

                results.append(item)
                pbar.update(1)

                if max_docs and len(results) >= max_docs:
                    return results

            # Check if there's a next page
            if not data.get("next"):
                break

            page += 1

    return results


def download_corpus(
    content_type: str,
    corpus_name: str,
    data_dir: Path,
    sources: list[str] | None = None,
    max_docs: int | None = None,
) -> None:
    """Download a corpus of content from Open5e.

    Args:
        content_type: Type of content ('monsters' or 'spells').
        corpus_name: Name for the corpus directory.
        data_dir: Base data directory.
        sources: Optional list of document slugs to filter by.
        max_docs: Maximum number of documents to download.
    """
    corpus_dir = data_dir / corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = corpus_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = corpus_dir / "metadata.json"

    # Load existing metadata for resume capability
    existing_slugs: set[str] = set()
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_slugs = set(existing_data.get("documents", {}).keys())
        logger.info(f"Found {len(existing_slugs)} existing documents")

    headers = {
        "User-Agent": "BiteSizeRAG-Corpus-Builder/1.0 (RPG evaluation corpus)",
    }

    # Select renderer based on content type
    render_func: Callable[[dict[str, Any]], str]
    if content_type == "monsters":
        render_func = render_monster_to_markdown
    elif content_type == "spells":
        render_func = render_spell_to_markdown
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    with httpx.Client(headers=headers, timeout=60.0) as client:
        logger.info(f"Fetching {content_type} from Open5e...")
        if sources:
            logger.info(f"Filtering by sources: {sources}")

        items = fetch_content(client, content_type, sources, max_docs)
        logger.info(f"Found {len(items)} items")

        # Render and save documents
        documents: dict[str, dict[str, Any]] = {}
        saved = 0
        skipped = 0

        for item in tqdm(items, desc="Rendering documents", unit="docs"):
            slug = item.get("slug", "")
            if not slug:
                continue

            # Skip if already exists
            md_path = docs_dir / f"{slug}.md"
            if md_path.exists():
                skipped += 1
                documents[slug] = {
                    "name": item.get("name", slug),
                    "source": item.get("document__slug", "unknown"),
                    "file": f"docs/{slug}.md",
                }
                continue

            # Render and save
            try:
                markdown = render_func(item)
                md_path.write_text(markdown, encoding="utf-8")
                saved += 1
                documents[slug] = {
                    "name": item.get("name", slug),
                    "source": item.get("document__slug", "unknown"),
                    "file": f"docs/{slug}.md",
                }
            except Exception as e:
                logger.warning(f"Failed to render {slug}: {e}")

        # Save metadata
        # Collect license info from sources
        source_counts: dict[str, int] = {}
        for doc in documents.values():
            src = doc["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

        metadata = {
            "corpus": corpus_name,
            "content_type": content_type,
            "sources": sources or ["all"],
            "total_documents": len(documents),
            "source_counts": source_counts,
            "api_base": OPEN5E_BASE_URL,
            "license": "Open Gaming License / Creative Commons (varies by source)",
            "documents": documents,
        }

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved: {saved}, Skipped: {skipped}")
        logger.info(f"Total documents in corpus: {len(documents)}")
        logger.info(f"Output directory: {corpus_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build RPG corpus from Open5e for RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python build_open5e.py monsters --source wotc-srd --corpus srd
    uv run python build_open5e.py spells --source wotc-srd --corpus srd_spells
    uv run python build_open5e.py monsters --source tob,cc --corpus kobold

Available sources (document slugs):
    wotc-srd     - D&D 5e SRD (CC-BY-4.0)
    tob          - Tome of Beasts (OGL)
    tob2         - Tome of Beasts 2 (OGL)
    tob3         - Tome of Beasts 3 (OGL)
    cc           - Creature Codex (OGL)
    dmag         - Deep Magic 5e (OGL)
    dmag-e       - Deep Magic Extended (OGL)
    menagerie    - Level Up A5e Monstrous Menagerie (OGL)
    a5e          - Level Up Advanced 5e (CC-BY-4.0)
    blackflag    - Black Flag SRD (ORC)
        """,
    )
    parser.add_argument(
        "content_type",
        choices=["monsters", "spells"],
        help="Type of content to download",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Comma-separated document slugs to filter (e.g., 'wotc-srd')",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Name for the corpus directory",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to download",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory path (default: ../data/)",
    )

    args = parser.parse_args()

    # Parse sources
    sources = None
    if args.source:
        sources = [s.strip() for s in args.source.split(",")]

    # Determine data directory
    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or (script_dir / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_corpus(
            content_type=args.content_type,
            corpus_name=args.corpus,
            data_dir=data_dir,
            sources=sources,
            max_docs=args.max_docs,
        )
        logger.info("Download complete!")

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user (Ctrl+C)")
        logger.info("Progress has been saved. Re-run to resume.")
        sys.exit(130)


if __name__ == "__main__":
    main()
