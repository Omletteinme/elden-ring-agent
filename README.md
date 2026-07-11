---
title: Elden Ring Agent
emoji: ⚔️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# Elden Ring Agent

An agent that answers Elden Ring questions by searching a real, current wiki corpus — not from an LLM's training data — and cites its sources. Built as an agentic tool-use system (not a static RAG pipeline): the agent decides when and what to search, and can make multiple search calls for questions that span multiple pages (e.g. "how has X changed").

## Why this exists

Most "chat with your docs" projects skip measurement. This one is built around an eval harness from day one: every retrieval and every answer is checked against hand-verified ground truth, and the results (including failures) are reported here, not hidden.

## Architecture

```
Wiki pages (scraped) → clean + chunk → embed → vector store
                                                      ↑
React chat UI → FastAPI /chat → Agent (Groq) → decides: answer directly, or call search_wiki tool (maybe more than once)
                                                      ↓
                                        Answer with citations to source chunks
```

- **Corpus**: Elden Ring wiki (Fandom), 150 pages fetched via the official MediaWiki API — see `scripts/scrape_wiki.py` and the "Design decisions" section below for why
- **Chunking**: infobox stats extracted as structured "Label: Value" chunks; body text split per-section, with page title + URL kept as metadata for citations — see `src/chunking.py`
- **Retrieval**: hybrid vector (sentence-transformers `all-MiniLM-L6-v2`, local/free) + keyword (BM25) search, fused via reciprocal rank fusion, with title-aware boosting when the query names a specific page — see `src/retrieval.py`
- **Vector store**: Chroma (local, persisted to `data/chroma/`)
- **Agent**: Groq (`openai/gpt-oss-20b`) with a `search_wiki` tool the model calls when it needs information, rather than always retrieving — see `src/agent.py`
- **Eval**: 27 hand-written, hand-verified QA pairs across 5 question types, scored on retrieval accuracy + answer correctness/faithfulness — see `eval/`

## Design decisions worth knowing (found the hard way)

- **MediaWiki API over scraping rendered pages.** Fextralife (the more detailed wiki) was down; Fandom's Cloudflare bot-management fingerprinted and blocked Python's `requests` client specifically on the rendered `/wiki/` route (identical headers worked via curl — a TLS/client fingerprinting issue, not a missing header). Rather than chase that with spoofed fingerprints or client-swapping, switched to Fandom's `api.php`, which `robots.txt` explicitly allowlists for all crawlers and returns clean article-body HTML with no site chrome to strip.
- **Title-aware boosting in retrieval.** A query mixing an entity name with a topic keyword (e.g. "Ancient Death Rancor patch") lost to *other* pages' more keyword-dense chunks — the correct page's own relevant chunk ranked 13th. Fixed by boosting any chunk whose title appears in the query, confirmed against the failing case.
- **Model choice for tool-calling.** `llama-3.3-70b-versatile` produced malformed tool-call syntax on this schema; `openai/gpt-oss-20b` handles it reliably, with one caveat below.
- **Gradio SDK, not Docker SDK, for the Hugging Face Spaces deploy.** Docker SDK requires a paid plan on this account. `app.py` at the repo root mounts the real FastAPI app (`src/api.py`, unchanged) onto a minimal Gradio page via `gr.mount_gradio_app` -- `/chat` and `/health` work exactly as before, verified locally; the Gradio page only exists to satisfy HF's free-tier SDK requirement.

## Status

Building phase by phase — see progress below.

- [x] Phase 0 — Project scaffold
- [x] Phase 1 — Corpus collection (150 pages: Weapons/Bosses/Talismans/Sorceries/Incantations, 30 each, via MediaWiki API)
- [x] Phase 2 — Clean + chunk (882 chunks from 150 pages)
- [x] Phase 3 — Embed + vector store (Chroma + BM25 hybrid retrieval, fused via RRF)
- [x] Phase 4 — Agent loop with tool-calling (Groq `openai/gpt-oss-20b`, multi-hop search)
- [x] Phase 5 — Eval harness (27 questions, see results below)
- [x] Phase 6 — Frontend + backend, running locally end-to-end (cloud deploy: not yet, see below)

## Eval results

27 hand-written QA pairs (`eval/qa_set.json`), ground truth pulled directly from the indexed chunks (not from memory, to keep it verifiably correct). Run with `python eval/run_eval.py`; full per-question results in `eval/results.json`.

| Metric | Score |
|---|---|
| Overall answer accuracy | 93% (25/27) |
| Retrieval full-hit rate (correct page(s) in top-5) | 100% |
| Factual questions (13) | 100% |
| Multi-hop / comparison questions (3) | 100% |
| Patch-history questions (3) | 100% |
| Out-of-scope refusal (5) | 100% |
| Adversarial near-miss names (3) | 33% |

**The adversarial result is the real finding here, not a footnote.** Given a question naming a plausible-but-nonexistent item close to a real one (e.g. "Ancient Death Ritual" vs. the actual "Ancient Death Rancor"), the agent fabricates a connection ("the spell commonly referred to as...") and answers using the real item's stats about 2/3 of the time, instead of declining. I added an explicit system-prompt rule against this ("never claim two differently-named things are related unless a search result says so") after finding it — it reduced but didn't eliminate the behavior. This is a known LLM failure mode (plausible-sounding fabrication under a near-miss), not something prompting alone reliably fixes, and it's exactly the kind of gap a real eval harness is supposed to surface rather than hide.

Two other things the eval process caught along the way (both fixed, not just noted):
- **A reliability bug**: `openai/gpt-oss-20b` occasionally leaks an internal format token into its tool-call name (`search_wiki<|channel|>commentary`), which Groq's validator rejects as a 400 — this crashed the whole answer with no retry. Added a retry wrapper (`_create_completion` in `src/agent.py`) since it's a sampling-level glitch, not deterministic.
- **A scoring bug in the eval harness itself**: the model outputs curly quotes (`'`) but my refusal-detection keywords used straight ASCII apostrophes, so every apostrophe-containing marker silently failed to match — several genuinely correct refusals were scored as failures until this was normalized. A reminder that the eval code needs as much scrutiny as the thing it's evaluating.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env    # add your GROQ_API_KEY

# one-time: build the corpus (or skip -- data/ is already committed)
python scripts/scrape_wiki.py
python src/chunking.py
python src/index.py
```

## Running it

```bash
# backend (from src/)
cd src && uvicorn api:app --port 8000 --reload

# frontend (from frontend/), in a separate terminal
cd frontend && npm install && npm run dev
```

Open the frontend's local URL (Vite prints it, typically `http://localhost:5173`). The backend must be running on port 8000 for the chat to work (`API_URL` in `frontend/src/App.jsx`).

**Not done yet: cloud deployment.** The app runs locally end-to-end (verified) but isn't deployed anywhere public. Deploying means creating accounts on a hosting provider (e.g. Render for the backend, Vercel for the frontend) and connecting them to this repo -- account creation and external service setup, so that's a deliberate choice to leave for you to do rather than something to do on your behalf. Render + Vercel both have straightforward free tiers if you want to do this next.

**Also not done: token streaming.** The agent loop runs to completion (including all tool-call rounds) before the API responds, so the frontend shows a "Searching and thinking…" state rather than streaming tokens in. Restructuring the tool-calling loop to yield partial output mid-round is real additional work; noted here rather than silently skipped.

## Project structure

```
data/
  raw/      # scraped wiki HTML (150 pages, committed)
  chunks/   # chunked + metadata (882 chunks, committed)
  chroma/   # vector store (gitignored, rebuild with src/index.py)
src/        # application code: chunking, indexing, retrieval, agent, FastAPI backend
scripts/    # one-off scripts (scraping, robots check)
eval/       # hand-written QA set + eval runner + results
frontend/   # React (Vite) chat UI
```
