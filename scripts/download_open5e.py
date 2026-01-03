#!/usr/bin/env python3
"""Download Open5e content from existing metadata.

Reads metadata.json and downloads content from Open5e API.
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

DELAY_SECONDS = 1.0
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0


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
    preamble: str = "",
) -> list[str]:
    """Render a block of actions with a heading."""
    if not actions:
        return []
    lines = [f"## {heading}", ""]
    if preamble:
        lines.extend([preamble, ""])
    for action in actions:
        name = action.get("name", "Action")
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
    """Render a monster JSON entry to markdown."""
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

    # Actions
    lines.extend(_render_action_block(monster.get("actions", []), "Actions"))
    lines.extend(
        _render_action_block(monster.get("bonus_actions", []), "Bonus Actions")
    )
    lines.extend(_render_action_block(monster.get("reactions", []), "Reactions"))
    lines.extend(
        _render_action_block(
            monster.get("legendary_actions", []),
            "Legendary Actions",
            monster.get("legendary_desc", ""),
        )
    )
    lines.extend(
        _render_action_block(monster.get("mythic_actions", []), "Mythic Actions")
    )
    lines.extend(_render_lair_actions(monster.get("lair_actions", [])))

    return "\n".join(lines)


def render_spell_to_markdown(spell: dict[str, Any]) -> str:
    """Render a spell JSON entry to markdown."""
    lines = []

    lines.extend([f"# {spell['name']}", ""])
    doc_title = spell.get("document__title", "Unknown")
    doc_slug = spell.get("document__slug", "unknown")
    lines.extend([f"**Source:** {doc_title} ({doc_slug})", ""])

    level = spell.get("level", "Cantrip")
    school = spell.get("school", "")
    ritual = spell.get("ritual", "no")
    level_line = f"*{level} {school}"
    if ritual == "yes":
        level_line += " (ritual)"
    level_line += "*"
    lines.extend([level_line, "", "---", ""])

    lines.append(f"**Casting Time:** {spell.get('casting_time', '1 action')}")
    lines.append(f"**Range:** {spell.get('range', 'Self')}")

    components = spell.get("components", "")
    material = spell.get("material", "")
    if material:
        lines.append(f"**Components:** {components} ({material})")
    else:
        lines.append(f"**Components:** {components}")

    lines.extend([f"**Duration:** {spell.get('duration', 'Instantaneous')}", ""])

    dnd_class = spell.get("dnd_class", "")
    if dnd_class:
        lines.extend([f"**Classes:** {dnd_class}", ""])

    lines.extend(["---", "", spell.get("desc", ""), ""])

    higher_level = spell.get("higher_level", "")
    if higher_level:
        lines.extend([f"**At Higher Levels.** {higher_level}", ""])

    return "\n".join(lines)


def fetch_item(
    client: httpx.Client,
    api_base: str,
    content_type: str,
    slug: str,
) -> dict[str, Any] | None:
    """Fetch a single item from Open5e API."""
    url = f"{api_base}/{content_type}/{slug}/"
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
    """Download content for a corpus from Open5e API.

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

    api_base = metadata.get("api_base", "https://api.open5e.com/v1")
    content_type = metadata.get("content_type", "monsters")
    documents: dict[str, dict[str, Any]] = metadata.get("documents", {})

    docs_dir = corpus_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Select renderer
    render_func: Callable[[dict[str, Any]], str]
    if content_type == "spells":
        render_func = render_spell_to_markdown
    else:
        render_func = render_monster_to_markdown

    items = list(documents.items())
    if max_docs is not None:
        items = items[:max_docs]

    print(f"Downloading {len(items)} documents to {docs_dir}")

    headers = {"User-Agent": "RAG-Corpus-Downloader/1.0"}
    failed = 0

    with httpx.Client(headers=headers, timeout=60.0) as client:
        for slug, doc_info in tqdm(items, desc="Downloading", unit="doc"):
            file_path = doc_info.get("file", f"docs/{slug}.md")
            output_path = corpus_dir / file_path

            if output_path.exists():
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            item = fetch_item(client, api_base, content_type, slug)
            if item:
                markdown = render_func(item)
                output_path.write_text(markdown, encoding="utf-8")
            else:
                failed += 1
                tqdm.write(f"Failed: {slug}")

            time.sleep(delay)

    if failed:
        print(f"Done ({failed} failed)")
    else:
        print("Done")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Open5e content from existing metadata"
    )
    parser.add_argument("corpus", help="Corpus name (e.g., dnd5e_srd_monsters)")
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
