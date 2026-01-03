#!/usr/bin/env python
"""Download Pathfinder 2e content from Archives of Nethys for RAG evaluation.

Fetches creatures, spells, feats, and other content from the Archives of Nethys
ElasticSearch backend, renders each entry to a markdown document, and saves to
a corpus directory.

Usage:
    uv run python download_aon.py creature --corpus pf2e_creatures --max-docs 500
    uv run python download_aon.py spell --corpus pf2e_spells --max-docs 500
    uv run python download_aon.py feat --corpus pf2e_feats --max-docs 500

Output:
    <data-dir>/<corpus>/
        metadata.json   - Corpus metadata with file counts and licensing
        docs/
            <id>.md     - Individual markdown documents

API Notes:
    - ElasticSearch endpoint: https://elasticsearch.aonprd.com/aon/_search
    - Categories: creature, spell, feat, action, ancestry, class, equipment, etc.
    - No authentication required
    - Content is under the ORC License
"""

import argparse
import contextlib
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

AON_ES_BASE_URL = "https://elasticsearch.aonprd.com/aon"

# Rate limiting - Archives of Nethys is a free community resource
# Be very conservative to avoid impacting the service
BASE_DELAY_SECONDS = 3.0  # Conservative delay between requests
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.5
MAX_BACKOFF_SECONDS = 300  # 5 minute max backoff
PAGE_SIZE = 50  # Smaller page size to reduce server load


def request_with_retry(
    client: httpx.Client,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    """Make HTTP request with exponential backoff on errors."""
    delay = BASE_DELAY_SECONDS
    last_exception: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            # Add jitter to avoid thundering herd
            # S311: random is fine here - jitter for rate limiting, not security
            jitter = random.uniform(0, delay * 0.1)  # noqa: S311
            sleep_time = delay + jitter
            logger.info(f"Retrying in {sleep_time:.1f}s (attempt {attempt + 1})")
            time.sleep(sleep_time)
            delay = min(delay * BACKOFF_FACTOR, MAX_BACKOFF_SECONDS)
        else:
            time.sleep(BASE_DELAY_SECONDS)

        try:
            if json_body:
                response = client.post(url, json=json_body)
            else:
                response = client.get(url)

            if response.status_code == 429:
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


def _add_field(lines: list[str], source: dict[str, Any], key: str, label: str) -> None:
    """Add a field to lines if present in source."""
    value = source.get(key, "")
    if value:
        lines.append(f"**{label}** {value}")


def _get_traits_str(source: dict[str, Any]) -> str:
    """Extract traits as comma-separated string."""
    traits = source.get("trait", [])
    if isinstance(traits, list):
        return ", ".join(traits) if traits else ""
    return str(traits)


def _render_creature_header(source: dict[str, Any]) -> list[str]:
    """Render creature header section."""
    lines = [f"# {source.get('name', 'Unknown Creature')}", ""]
    lines.extend([f"**Source:** {source.get('source', 'Unknown')}", ""])
    lines.append(f"*{source.get('type', 'Creature')} {source.get('level', 0)}*")
    trait_str = _get_traits_str(source)
    if trait_str:
        lines.append(f"**Traits** {trait_str}")
    lines.append("")
    return lines


def _render_creature_stats(source: dict[str, Any]) -> list[str]:
    """Render creature perception, senses, languages, skills."""
    lines = ["---", ""]
    for key, label in [
        ("perception", "Perception"),
        ("sense", "Senses"),
        ("language", "Languages"),
        ("skill", "Skills"),
        ("abilityMod", "Ability Modifiers"),
    ]:
        _add_field(lines, source, key, label)
    lines.append("")
    if source.get("item"):
        lines.extend([f"**Items** {source['item']}", ""])
    return lines


def _render_creature_defenses(source: dict[str, Any]) -> list[str]:
    """Render creature AC, saves, HP, immunities, resistances, weaknesses."""
    lines = ["---", ""]
    _add_field(lines, source, "ac", "AC")

    saves = []
    for key, name in [("fort", "Fort"), ("ref", "Ref"), ("will", "Will")]:
        if source.get(key):
            saves.append(f"{name} {source[key]}")
    if saves:
        lines.append(f"**Saving Throws** {', '.join(saves)}")

    for key, label in [
        ("hp", "HP"),
        ("immunity", "Immunities"),
        ("resistance", "Resistances"),
        ("weakness", "Weaknesses"),
    ]:
        _add_field(lines, source, key, label)
    lines.append("")
    if source.get("speed"):
        lines.extend([f"**Speed** {source['speed']}", ""])
    return lines


def render_creature_to_markdown(creature: dict[str, Any]) -> str:
    """Render a PF2e creature entry to markdown."""
    source = creature.get("_source", creature)
    lines: list[str] = []

    lines.extend(_render_creature_header(source))
    lines.extend(_render_creature_stats(source))
    lines.extend(_render_creature_defenses(source))

    lines.extend(["---", ""])
    text = source.get("text", "")
    if text and isinstance(text, str):
        lines.extend([re.sub(r"<[^>]+>", "", text), ""])

    markdown = source.get("markdown", "")
    if markdown:
        lines.extend([markdown, ""])

    return "\n".join(lines)


def _render_entry_header(
    source: dict[str, Any], default_name: str, default_type: str
) -> list[str]:
    """Render common header for spells and feats."""
    lines = [f"# {source.get('name', default_name)}", ""]
    lines.extend([f"**Source:** {source.get('source', 'Unknown')}", ""])
    lines.append(f"*{source.get('type', default_type)} {source.get('level', 0)}*")
    trait_str = _get_traits_str(source)
    if trait_str:
        lines.append(f"**Traits** {trait_str}")
    lines.append("")
    return lines


def render_spell_to_markdown(spell: dict[str, Any]) -> str:
    """Render a PF2e spell entry to markdown."""
    source = spell.get("_source", spell)
    lines: list[str] = _render_entry_header(source, "Unknown Spell", "Spell")

    for key, label in [
        ("tradition", "Traditions"),
        ("cast", "Cast"),
        ("component", "Components"),
        ("range", "Range"),
        ("area", "Area"),
        ("target", "Targets"),
        ("duration", "Duration"),
        ("save", "Saving Throw"),
    ]:
        _add_field(lines, source, key, label)

    lines.extend(["", "---", ""])

    text = source.get("text", "")
    if text and isinstance(text, str):
        lines.extend([re.sub(r"<[^>]+>", "", text), ""])

    heightened = source.get("heightened", "")
    if heightened:
        lines.extend(["**Heightened**", heightened, ""])

    return "\n".join(lines)


def render_feat_to_markdown(feat: dict[str, Any]) -> str:
    """Render a PF2e feat entry to markdown."""
    source = feat.get("_source", feat)
    lines = []

    name = source.get("name", "Unknown Feat")
    lines.append(f"# {name}")
    lines.append("")

    # Source info
    src_book = source.get("source", "Unknown")
    lines.append(f"**Source:** {src_book}")
    lines.append("")

    # Feat level and traits
    level = source.get("level", 0)
    feat_type = source.get("type", "Feat")
    traits = source.get("trait", [])
    if isinstance(traits, list):
        trait_str = ", ".join(traits) if traits else ""
    else:
        trait_str = str(traits)

    lines.append(f"*{feat_type} {level}*")
    if trait_str:
        lines.append(f"**Traits** {trait_str}")
    lines.append("")

    # Prerequisites
    prereqs = source.get("prerequisite", "")
    if prereqs:
        lines.append(f"**Prerequisites** {prereqs}")

    # Requirements
    reqs = source.get("requirement", "")
    if reqs:
        lines.append(f"**Requirements** {reqs}")

    # Trigger
    trigger = source.get("trigger", "")
    if trigger:
        lines.append(f"**Trigger** {trigger}")

    # Frequency
    frequency = source.get("frequency", "")
    if frequency:
        lines.append(f"**Frequency** {frequency}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Description
    text = source.get("text", "")
    if text and isinstance(text, str):
        clean_text = re.sub(r"<[^>]+>", "", text)
        lines.append(clean_text)
        lines.append("")

    return "\n".join(lines)


def fetch_content(
    client: httpx.Client,
    category: str,
    max_docs: int,
) -> list[dict[str, Any]]:
    """Fetch content from Archives of Nethys ElasticSearch."""
    results: list[dict[str, Any]] = []
    from_offset = 0

    search_url = f"{AON_ES_BASE_URL}/_search"

    with tqdm(total=max_docs, desc=f"Fetching {category}s", unit="items") as pbar:
        while len(results) < max_docs:
            query = {
                "query": {
                    "match": {
                        "category": category,
                    },
                },
                "size": min(PAGE_SIZE, max_docs - len(results)),
                "from": from_offset,
            }

            try:
                response = request_with_retry(client, search_url, json_body=query)
                data = response.json()
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch: {e}")
                break

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                if len(results) >= max_docs:
                    break
                results.append(hit)
                pbar.update(1)

            from_offset += len(hits)

            # Check if we've retrieved all available
            total = data.get("hits", {}).get("total", {})
            total_count = total.get("value", 0) if isinstance(total, dict) else total

            if from_offset >= total_count:
                break

    return results


def download_corpus(
    category: str,
    corpus_name: str,
    data_dir: Path,
    max_docs: int,
) -> None:
    """Download a corpus from Archives of Nethys.

    Args:
        category: Content category ('creature', 'spell', 'feat', etc.).
        corpus_name: Name for the corpus directory.
        data_dir: Base data directory.
        max_docs: Maximum number of documents to download.
    """
    corpus_dir = data_dir / corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = corpus_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = corpus_dir / "metadata.json"

    # Select renderer based on category
    renderers = {
        "creature": render_creature_to_markdown,
        "spell": render_spell_to_markdown,
        "feat": render_feat_to_markdown,
    }

    render_func = renderers.get(category)
    if not render_func:
        # Use a generic renderer for other categories
        render_func = render_feat_to_markdown  # Generic enough for most
        logger.warning(f"Using generic renderer for category: {category}")

    headers = {
        "User-Agent": "BiteSizeRAG-Corpus-Builder/1.0 (RPG evaluation corpus)",
        "Content-Type": "application/json",
    }

    with httpx.Client(headers=headers, timeout=60.0) as client:
        logger.info(f"Fetching {category}s from Archives of Nethys...")

        items = fetch_content(client, category, max_docs)
        logger.info(f"Found {len(items)} items")

        # Render and save documents
        documents: dict[str, dict[str, Any]] = {}
        saved = 0
        skipped = 0

        for item in tqdm(items, desc="Rendering documents", unit="docs"):
            # Get document ID
            doc_id = item.get("_id", "")
            if not doc_id:
                continue

            # Create safe filename from ID
            safe_id = doc_id.replace("/", "_").replace("\\", "_")
            md_path = docs_dir / f"{safe_id}.md"

            # Skip if already exists
            if md_path.exists():
                skipped += 1
                source = item.get("_source", {})
                documents[safe_id] = {
                    "name": source.get("name", safe_id),
                    "source": source.get("source", "Unknown"),
                    "file": f"docs/{safe_id}.md",
                }
                continue

            # Render and save
            try:
                markdown = render_func(item)
                md_path.write_text(markdown, encoding="utf-8")
                saved += 1
                source = item.get("_source", {})
                documents[safe_id] = {
                    "name": source.get("name", safe_id),
                    "source": source.get("source", "Unknown"),
                    "file": f"docs/{safe_id}.md",
                }
            except Exception as e:
                logger.warning(f"Failed to render {doc_id}: {e}")

        # Save metadata
        metadata = {
            "corpus": corpus_name,
            "category": category,
            "total_documents": len(documents),
            "api_base": AON_ES_BASE_URL,
            "license": "ORC License (Open RPG Creative License)",
            "license_url": "https://paizo.com/orclicense",
            "attribution": (
                "Content from Archives of Nethys (2e.aonprd.com), "
                "Pathfinder 2e by Paizo Inc."
            ),
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
        description="Download PF2e content from Archives of Nethys for RAG eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python download_aon.py creature --corpus pf2e_creatures --max-docs 500
    uv run python download_aon.py spell --corpus pf2e_spells --max-docs 500
    uv run python download_aon.py feat --corpus pf2e_feats --max-docs 500

Available categories:
    creature     - Monsters and NPCs
    spell        - Spells and cantrips
    feat         - Feats and abilities
    action       - Actions
    ancestry     - Ancestries (races)
    class        - Character classes
    equipment    - Items and gear
    hazard       - Traps and hazards
        """,
    )
    parser.add_argument(
        "category",
        type=str,
        help="Content category to download (creature, spell, feat, etc.)",
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
        default=500,
        help="Maximum documents to download (default: 500)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory path (default: ../data/)",
    )

    args = parser.parse_args()

    # Determine data directory
    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or (script_dir / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_corpus(
            category=args.category,
            corpus_name=args.corpus,
            data_dir=data_dir,
            max_docs=args.max_docs,
        )
        logger.info("Download complete!")

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user (Ctrl+C)")
        logger.info("Progress has been saved. Re-run to resume.")
        sys.exit(130)


if __name__ == "__main__":
    main()
