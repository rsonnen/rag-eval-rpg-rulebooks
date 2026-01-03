# rag-eval-rpg-rulebooks

Evaluation corpus of tabletop RPG content for testing RAG systems.

## What This Is

This repository contains **evaluation data for RAG systems**:

- **corpus.yaml** - Evaluation scenarios for each corpus
- **metadata.json** - Document inventory
- **Generated questions** - Validated Q/A pairs (where available)

The actual content (markdown documents) is not included. Use the download scripts to fetch from the APIs.

## Available Corpora

### Open5e (D&D 5e)

| Corpus | Documents | Description |
|--------|-----------|-------------|
| `dnd5e_srd_monsters` | 322 | D&D 5e SRD monsters |
| `dnd5e_srd_spells` | 319 | D&D 5e SRD spells |
| `kobold_press_monsters` | 1,527 | Tome of Beasts + Creature Codex |
| `deep_magic_spells` | 578 | Deep Magic spells |

### Archives of Nethys (Pathfinder 2e)

| Corpus | Documents | Description |
|--------|-----------|-------------|
| `pf2e_creatures` | 500 | Pathfinder 2e creatures |
| `pf2e_spells` | 500 | Pathfinder 2e spells |
| `pf2e_feats` | 500 | Pathfinder 2e feats |

## Quick Start

```bash
cd scripts
uv sync

# Download Open5e content
uv run python download_open5e.py dnd5e_srd_monsters --max-docs 10

# Download Archives of Nethys content
uv run python download_aon.py pf2e_creatures --max-docs 10
```

## Directory Structure

```
<corpus>/
    corpus.yaml         # Evaluation configuration
    metadata.json       # Document inventory
    docs/               # Markdown documents (gitignored)

scripts/
    download_open5e.py  # Download from Open5e metadata
    download_aon.py     # Download from Archives of Nethys metadata
    build_open5e.py     # Build new Open5e corpora
    build_aon.py        # Build new Archives of Nethys corpora
```

## Metadata Format

### Open5e

```json
{
  "corpus": "dnd5e_srd_monsters",
  "content_type": "monsters",
  "sources": ["wotc-srd"],
  "total_documents": 322,
  "api_base": "https://api.open5e.com/v1",
  "documents": {
    "aboleth": {
      "name": "Aboleth",
      "source": "wotc-srd",
      "file": "docs/aboleth.md"
    }
  }
}
```

### Archives of Nethys

```json
{
  "corpus": "pf2e_creatures",
  "category": "creature",
  "total_documents": 500,
  "api_base": "https://elasticsearch.aonprd.com/aon",
  "documents": {
    "creature-1": {
      "name": "Unseen Servant",
      "source": ["Core Rulebook"],
      "file": "docs/creature-1.md"
    }
  }
}
```

## Building New Corpora

```bash
cd scripts

# Build Open5e corpus
uv run python build_open5e.py monsters --source wotc-srd --corpus my_monsters

# Build Archives of Nethys corpus
uv run python build_aon.py creature --corpus my_creatures --max-docs 100
```
