# Elden Ring Agent

An agent that answers Elden Ring questions by searching a real, current wiki corpus — not from an LLM's training data — and cites its sources. Built as an agentic tool-use system (not a static RAG pipeline): the agent decides when and what to search, and can make multiple search calls for questions that span multiple pages (e.g. "how has X changed").

## Why this exists

Most "chat with your docs" projects skip measurement. This one is built around an eval harness from day one: every retrieval and every answer is checked against hand-verified ground truth, and the results (including failures) are reported here, not hidden.

## Architecture

```
Wiki pages (scraped) → clean + chunk → embed → vector store
                                                      ↑
User question → Agent (Groq/Llama) → decides: answer directly, or call search_wiki tool (maybe more than once)
                                                      ↓
                                        Answer with citations to source chunks
```

- **Corpus**: Elden Ring wiki (Fextralife), scraped politely (robots.txt-respecting, rate-limited) — see `scripts/scrape_wiki.py`
- **Chunking**: per-section, with page title + URL kept as metadata for citations
- **Embeddings**: sentence-transformers (local, free, CPU-friendly)
- **Vector store**: Chroma (local) — see Phase 3
- **Agent**: Groq (Llama) with a `search_wiki` tool the model calls when it needs information, rather than always retrieving
- **Eval**: hand-written QA set, scored on retrieval accuracy + answer faithfulness (see `eval/`)

## Status

Building phase by phase — see progress below.

- [x] Phase 0 — Project scaffold
- [x] Phase 1 — Corpus collection (150 pages: Weapons/Bosses/Talismans/Sorceries/Incantations, 30 each, via MediaWiki API)
- [x] Phase 2 — Clean + chunk (882 chunks from 150 pages)
- [x] Phase 3 — Embed + vector store (Chroma + BM25 hybrid retrieval, fused via RRF)
- [x] Phase 4 — Agent loop with tool-calling (Groq `openai/gpt-oss-20b`, multi-hop search)
- [ ] Phase 5 — Eval harness
- [ ] Phase 6 — Frontend + deploy

## Eval results

_Filled in once Phase 5 is done._

| Metric | Score |
|---|---|
| Retrieval accuracy (right chunk retrieved) | TBD |
| Answer faithfulness (no unsupported claims) | TBD |

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env    # add your GROQ_API_KEY
```

## Project structure

```
data/
  raw/      # scraped wiki HTML (gitignored)
  clean/    # cleaned text
  chunks/   # chunked + metadata, ready to embed
src/        # application code (chunking, embedding, agent, API)
scripts/    # one-off scripts (scraping, robots check, indexing)
eval/       # hand-written QA set + eval runner
tests/
```
