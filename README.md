# rag-eval-rpg-rulebooks

Evaluation corpus of tabletop RPG content (monsters, spells, feats) for testing RAG systems with complex structured documents.

## What This Is

This repository contains **evaluation data for RAG systems**:

- **corpus.yaml** - Evaluation configuration defining domain context and testing scenarios
- **Generated questions** - Validated Q/A pairs for evaluation (where available)
- **metadata.json** - Document inventory with source info
- **Download scripts** - Fetch content from Open5e and Archives of Nethys

The actual content (markdown documents) is not included - use the download scripts to fetch from the APIs.

## Purpose

This corpus tests document-processor handling of:
- Complex tables (stat blocks with abilities, attacks, saves)
- Nested hierarchical content (actions, reactions, legendary actions)
- Dense cross-references (spell references, ability interactions)
- Game-specific formatting (traits, conditions, damage types)

## Available Sources

### Open5e API (D&D 5e ecosystem)

| Source Slug | Content | License | Est. Count |
|-------------|---------|---------|------------|
| wotc-srd | D&D 5e SRD | CC-BY-4.0 | ~300 monsters, ~300 spells |
| tob | Tome of Beasts | OGL | ~400 monsters |
| tob2 | Tome of Beasts 2 | OGL | ~400 monsters |
| tob3 | Tome of Beasts 3 | OGL | ~400 monsters |
| cc | Creature Codex | OGL | ~400 monsters |
| dmag | Deep Magic 5e | OGL | ~300 spells |
| dmag-e | Deep Magic Extended | OGL | ~200 spells |
| menagerie | Level Up A5e Menagerie | OGL | ~500 monsters |
| a5e | Level Up Advanced 5e | CC-BY-4.0 | Various |
| blackflag | Black Flag SRD | ORC | Various |

### Archives of Nethys (Pathfinder 2e)

| Category | Content | License | Est. Count |
|----------|---------|---------|------------|
| creature | Monsters and NPCs | ORC | ~4000 |
| spell | Spells and cantrips | ORC | ~2400 |
| feat | Feats and abilities | ORC | ~3000+ |
| action | Actions | ORC | ~500 |
| ancestry | Ancestries (races) | ORC | ~50 |
| class | Character classes | ORC | ~20 |
| equipment | Items and gear | ORC | ~2000 |
| hazard | Traps and hazards | ORC | ~300 |

## Usage

### Install dependencies

```bash
cd scripts/corpus/rpg_rulebooks
uv sync
```

### Download from Open5e

```bash
# D&D 5e SRD monsters
uv run python download_open5e.py monsters --source wotc-srd --corpus dnd5e_srd_monsters

# D&D 5e SRD spells
uv run python download_open5e.py spells --source wotc-srd --corpus dnd5e_srd_spells

# Kobold Press monsters (Tome of Beasts series + Creature Codex)
uv run python download_open5e.py monsters --source tob,tob2,tob3,cc --corpus kobold_press_monsters

# Deep Magic spells
uv run python download_open5e.py spells --source dmag,dmag-e --corpus deep_magic_spells
```

### Download from Archives of Nethys

```bash
# Pathfinder 2e creatures
uv run python download_aon.py creature --corpus pf2e_creatures --max-docs 500

# Pathfinder 2e spells
uv run python download_aon.py spell --corpus pf2e_spells --max-docs 500

# Pathfinder 2e feats
uv run python download_aon.py feat --corpus pf2e_feats --max-docs 500
```

## Output Structure

```
<corpus_name>/
    corpus.yaml             # Evaluation configuration
    metadata.json           # Corpus metadata
    docs/                   # Downloaded documents (gitignored)
        <slug>.md           # Individual markdown documents
```

### Example Monster Document

```markdown
# Aboleth

**Source:** 5e Core Rules (wotc-srd)

*Large Aberration, lawful evil*

---

**Armor Class** 17 (natural armor)
**Hit Points** 135 (18d10+36)
**Speed** 10 ft., swim 40 ft.

---

| STR | DEX | CON | INT | WIS | CHA |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 21 (+5) | 9 (-1) | 15 (+2) | 18 (+4) | 15 (+2) | 18 (+4) |

---

**Saving Throws** CON +6, INT +8, WIS +6
**Skills** History +12, Perception +10
**Senses** darkvision 120 ft., passive Perception 20
**Languages** Deep Speech, telepathy 120 ft.
**Challenge** 10

---

***Amphibious.*** The aboleth can breathe air and water.

## Actions

***Multiattack.*** The aboleth makes three tentacle attacks.

***Tentacle.*** Melee Weapon Attack: +9 to hit, reach 10 ft...
```

### Metadata Schema

```json
{
  "corpus": "dnd5e_srd_monsters",
  "content_type": "monsters",
  "sources": ["wotc-srd"],
  "total_documents": 325,
  "source_counts": {"wotc-srd": 325},
  "api_base": "https://api.open5e.com/v1",
  "license": "Open Gaming License / Creative Commons (varies by source)",
  "documents": {
    "aboleth": {"name": "Aboleth", "source": "wotc-srd", "file": "aboleth.md"}
  }
}
```

## Licensing

### Open Gaming License (OGL)

Most D&D 5e third-party content (Kobold Press, etc.) is published under the Open Gaming License. The OGL allows use of game mechanics but not product identity (setting-specific names, characters, etc.).

### Creative Commons Attribution 4.0 (CC-BY-4.0)

The D&D 5e SRD 5.1 was released under CC-BY-4.0 in January 2023. This allows sharing and adaptation with attribution.

**Attribution:** "This work includes material from the System Reference Document 5.1 (SRD 5.1) by Wizards of the Coast LLC, licensed under the Creative Commons Attribution 4.0 International License."

### ORC License (Open RPG Creative License)

Pathfinder 2e content from Paizo is published under the ORC License, a system-agnostic open gaming license.

**Attribution:** "This work includes content from Pathfinder 2nd Edition by Paizo Inc., available at Archives of Nethys (2e.aonprd.com) under the ORC License."

## Rate Limiting

Both scripts implement conservative rate limiting (3 second delays between requests) to respect these free community services. Do not modify these limits.

## Notes

- Downloads are resumable - re-running skips existing files
- Scripts generate progress bars and logging output
- All content rendered to markdown with proper stat block formatting
- Cross-references preserved where possible
