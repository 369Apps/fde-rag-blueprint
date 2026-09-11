# fde-rag-blueprint

A production-shaped RAG service template — the way I'd ship retrieval at a client engagement. Hybrid **vector + graph** retrieval, an eval harness, request tracing, and Docker packaging. Runs in demo mode with zero infrastructure; point it at Postgres + Neo4j when you're serious.

Built as a portfolio piece for Forward Deployed Engineer work: the boring parts (evals, tracing, config) are the point.

## Quickstart (demo mode, no infra)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Ingest and ask:

```bash
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"documents": [{"id": "doc1", "text": "Refunds are issued within 14 days of purchase."}]}'

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How long do refunds take?"}'
```

In demo mode the generator echoes retrieved context with citations instead of calling an LLM. Set `OPENAI_API_KEY` to get real generation.

## Production mode

Set env vars and the same code paths switch to real infrastructure:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Real answer generation |
| `DATABASE_URL` | Postgres + pgvector for vector store |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph store for entity relations |

## Layout

```
app/
  main.py        # FastAPI app: /ingest, /ask, /health, /metrics
  retrieval.py   # Hybrid retriever: vector search + graph expansion + rerank
  stores.py      # Storage backends: in-memory (demo), pgvector, Neo4j
  generation.py  # Answer generator: demo echo or OpenAI, always cited
  evals.py       # Eval harness: faithfulness, citation coverage, latency
  tracing.py     # Request tracing middleware + structured logs
```

## Evals

```bash
python -m app.evals
```

Runs the golden question set against the pipeline and reports faithfulness (every claim cited), citation coverage, and p95 latency. Gate deployments on this — a RAG system without evals is a demo.

## Why this shape

Client RAG pilots usually die on: no evals, no observability, retrieval that can't explain itself. This template bakes all three in from commit one, so the pilot and the production system are the same codebase.
