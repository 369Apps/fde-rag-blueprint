"""FastAPI service: /ingest, /ask, /explain, /health, /metrics."""
from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel

from .generation import generate
from .retrieval import HybridRetriever
from .stores import Document, default_stores
from .tracing import TracingMiddleware

app = FastAPI(title="fde-rag-blueprint", version="0.1.0")
app.add_middleware(TracingMiddleware)

_doc_store, _graph_store = default_stores()
retriever = HybridRetriever(_doc_store, _graph_store)
_started_at = time.time()
_ask_count = 0
_latencies: list[float] = []


class IngestRequest(BaseModel):
    documents: list[dict]


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - _started_at, 1)}


@app.get("/metrics")
def metrics():
    import statistics

    return {
        "asks": _ask_count,
        "latency_ms_p50": round(statistics.median(_latencies), 1) if _latencies else 0,
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    docs = [
        Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))
        for d in req.documents
    ]
    n = retriever.ingest(docs)
    return {"ingested": n}


@app.post("/ask")
def ask(req: AskRequest):
    global _ask_count
    start = time.perf_counter()
    results = retriever.retrieve(req.question)
    out = generate(req.question, results)
    _ask_count += 1
    _latencies.append((time.perf_counter() - start) * 1000)
    return out


@app.post("/explain")
def explain(req: AskRequest):
    """Show the retrieval trace: what was retrieved, boosted, and why."""
    return retriever.explain(req.question)
