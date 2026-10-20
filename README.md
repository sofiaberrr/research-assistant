# Agentic Research Assistant

An agentic LLM app that answers questions about 220 recent arXiv quantum computing papers. Built on the Claude API with LangGraph, it decides on its own which tool to use (full-text search, SQL over paper metadata, or web search) and cites its sources down to the page.

**Stack:** Python, Claude API, LangGraph, Chroma, SQLite, FastAPI, Docker, pytest

## Example

```
Q: What is a surface code and why is it popular?

Tools used:
  -> search_papers("surface code definition quantum error correction")
  -> search_papers("why surface code is popular: high threshold, nearest-neighbor 2D connectivity")

A: The surface code is a quantum error-correcting code that protects quantum information
from bit-flip and phase-flip errors [arXiv:2610.04613, p.1]. It's popular because it needs
only nearest-neighbor interactions on a planar square lattice [arXiv:2610.04613, p.1], has a
high threshold of about 0.01 [arXiv:2610.04613, p.1], and allows efficient decoding with
minimum-weight perfect matching [arXiv:2610.04613, p.1] ...
```

```
Q: How many papers in the collection were published in October 2026?

Tools used:
  -> query_database("SELECT COUNT(*) FROM papers WHERE published_date >= '2026-10-01' AND published_date < '2026-11-01'")

A: 170 papers.
```

## How it works

```
                 ┌──────────────┐
question ──────> │    agent     │ ──── done ────> answer + citations
                 │   (Claude)   │
                 └──────┬───────┘
                   tool │ ▲ results
                        ▼ │
   ┌──────────────┬──────────────┬──────────────┐
   │ search_papers│query_database│  web_search  │
   │ (Chroma RAG) │   (SQLite)   │  (fallback)  │
   └──────────────┴──────────────┴──────────────┘
```

- **Ingestion** (`ingest/`): downloads 220 papers from the arXiv API, stores metadata in SQLite, and splits PDF text into overlapping 1,000-character chunks embedded in Chroma. Each chunk keeps its paper ID and page number for citations. Bibliography chunks are filtered out, since they're keyword-dense and were crowding out real content in early retrieval tests.
- **Agent** (`agent/`): a two-node LangGraph loop. Claude picks tools based on their docstrings, and can call several in sequence (e.g. search the papers, find nothing, fall back to the web). The SQL tool opens the database read-only and rejects non-SELECT queries.
- **API** (`api/`): FastAPI with 4 endpoints, request validation via Pydantic, and auto-generated docs at `/docs`.

## Evaluation

A 40-question benchmark (`eval/benchmark.json`): 30 questions generated from random passages across 30 different papers (so the source paper is known), plus 10 metadata questions with SQL-computed ground truth. Answers are scored by an LLM judge.

| Run | Change | Accuracy | Groundedness | Source hit rate | Avg latency |
|---|---|---|---|---|---|
| baseline | | 95% | 74% | 97% | 5.9s |
| v2 | Stricter grounding prompt; judge sees full context + tool calls | 85% | 98% | 97% | 5.1s |
| v3 | Query reformulation on misses; retrieve 8 passages instead of 5 | **90%** | **100%** | **97%** | **5.2s** |

What I learned from iterating:
- **The baseline's accuracy was partly hiding a problem.** A quarter of its answers included claims not found in the retrieved text (extra derivations, equation numbers, background from the model's own knowledge).
- **Fixing groundedness exposed a retrieval problem.** Once the agent stopped filling gaps from memory, it gave up after one unsuccessful search. Telling it to rephrase and retry, and widening retrieval, recovered most of the accuracy.
- **Evals have bugs too.** The v1 judge only saw a truncated slice of the context, so it flagged some correct citations as unsupported. Part of the v2 gain comes from fixing that.

Full per-question results are in `eval/results/`.

## Run it yourself

Requires Python 3.12 and an [Anthropic API key](https://console.anthropic.com).

```bash
pip install -r requirements.txt
cp .env.example .env              # add your API key

python -m ingest.fetch_papers     # download papers (~15 min)
python -m ingest.build_index      # build the vector index (~15 min)

python -m agent.graph "What is a surface code?"   # ask from the command line
uvicorn api.main:app --reload                     # or serve the API: localhost:8000/docs
```

**Docker:**
```bash
docker build -t research-assistant .
docker run -p 8000:8000 --env-file .env research-assistant
```

**Tests and eval:**
```bash
python -m pytest -v               # 16 tests: chunking, filtering, tool safety, all endpoints
python -m eval.run_eval my_run    # score the agent on the benchmark
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ask` | Ask a question; returns the answer, tools used, and latency |
| GET | `/papers?year=&limit=` | List papers |
| GET | `/papers/{paper_id}` | Paper details and abstract |
