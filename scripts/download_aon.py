#!/usr/bin/env python3
"""Download Archives of Nethys content from existing metadata.

Reads metadata.json and downloads content from Archives of Nethys ElasticSearch.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

DELAY_SECONDS = 1.0
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0


def _get_traits_str(source: dict[str, Any]) -> str:
    """Extract traits as comma-separated string."""
    traits = source.get("trait", [])
    if isinstance(traits, list):
        return ", ".join(traits) if traits else ""
    return str(traits)


def _add_field(lines: list[str], source: dict[str, Any], key: str, label: str) -> None:
    """Add a field to lines if present in source."""
    value = source.get(key, "")
    if value:
        lines.append(f"**{label}** {value}")


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


def render_creature_to_markdown(item: dict[str, Any]) -> str:
    """Render a PF2e creature entry to markdown."""
    source = item.get("_source", item)
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


def render_spell_to_markdown(item: dict[str, Any]) -> str:
    """Render a PF2e spell entry to markdown."""
    source = item.get("_source", item)
    lines: list[str] = []

    lines.extend([f"# {source.get('name', 'Unknown Spell')}", ""])
    lines.extend([f"**Source:** {source.get('source', 'Unknown')}", ""])
    lines.append(f"*{source.get('type', 'Spell')} {source.get('level', 0)}*")
    trait_str = _get_traits_str(source)
    if trait_str:
        lines.append(f"**Traits** {trait_str}")
    lines.append("")

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


def render_feat_to_markdown(item: dict[str, Any]) -> str:
    """Render a PF2e feat entry to markdown."""
    source = item.get("_source", item)
    lines: list[str] = []

    lines.extend([f"# {source.get('name', 'Unknown Feat')}", ""])
    lines.extend([f"**Source:** {source.get('source', 'Unknown')}", ""])

    level = source.get("level", 0)
    feat_type = source.get("type", "Feat")
    lines.append(f"*{feat_type} {level}*")

    trait_str = _get_traits_str(source)
    if trait_str:
        lines.append(f"**Traits** {trait_str}")
    lines.append("")

    for key, label in [
        ("prerequisite", "Prerequisites"),
        ("requirement", "Requirements"),
        ("trigger", "Trigger"),
        ("frequency", "Frequency"),
    ]:
        _add_field(lines, source, key, label)

    lines.extend(["", "---", ""])

    text = source.get("text", "")
    if text and isinstance(text, str):
        lines.extend([re.sub(r"<[^>]+>", "", text), ""])

    return "\n".join(lines)


def fetch_item(
    client: httpx.Client,
    api_base: str,
    doc_id: str,
) -> dict[str, Any] | None:
    """Fetch a single item from Archives of Nethys ElasticSearch."""
    url = f"{api_base}/_doc/{doc_id}"
    delay = DELAY_SECONDS

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            jitter = random.uniform(0, delay * 0.1)  # noqa: S311
            time.sleep(delay + jitter)
            delay = min(delay * BACKOFF_FACTOR, 30.0)
        else:
            time.sleep(DELAY_SECONDS)

        try:
            response = client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPError:
            continue

    return None


def download_corpus(
    corpus_dir: Path,
    delay: float = 1.0,
    max_docs: int | None = None,
) -> None:
    """Download content for a corpus from Archives of Nethys.

    Args:
        corpus_dir: Path to corpus directory containing metadata.json.
        delay: Additional delay between requests.
        max_docs: Maximum number of documents to download.
    """
    metadata_path = corpus_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found", file=sys.stderr)
        sys.exit(1)

    with metadata_path.open(encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    api_base = metadata.get("api_base", "https://elasticsearch.aonprd.com/aon")
    category = metadata.get("category", "creature")
    documents: dict[str, dict[str, Any]] = metadata.get("documents", {})

    docs_dir = corpus_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Select renderer based on category
    renderers = {
        "creature": render_creature_to_markdown,
        "spell": render_spell_to_markdown,
        "feat": render_feat_to_markdown,
    }
    render_func = renderers.get(category, render_feat_to_markdown)

    items = list(documents.items())
    if max_docs is not None:
        items = items[:max_docs]

    print(f"Downloading {len(items)} documents to {docs_dir}")

    headers = {
        "User-Agent": "RAG-Corpus-Downloader/1.0",
        "Content-Type": "application/json",
    }
    failed = 0

    with httpx.Client(headers=headers, timeout=60.0) as client:
        for doc_id, doc_info in tqdm(items, desc="Downloading", unit="doc"):
            file_path = doc_info.get("file", f"docs/{doc_id}.md")
            output_path = corpus_dir / file_path

            if output_path.exists():
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            item = fetch_item(client, api_base, doc_id)
            if item:
                markdown = render_func(item)
                output_path.write_text(markdown, encoding="utf-8")
            else:
                failed += 1
                tqdm.write(f"Failed: {doc_id}")

            time.sleep(delay)

    if failed:
        print(f"Done ({failed} failed)")
    else:
        print("Done")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Archives of Nethys content from existing metadata"
    )
    parser.add_argument("corpus", help="Corpus name (e.g., pf2e_creatures)")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to download (default: all)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    corpus_dir = repo_root / args.corpus

    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    download_corpus(corpus_dir, args.delay, args.max_docs)


if __name__ == "__main__":
    main()
