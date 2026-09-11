"""Eval harness: golden question set + faithfulness / coverage / latency.

Run:  python -m app.evals
Gate deployments on this. A RAG pipeline that can't pass its own evals
should not be taking customer questions.
"""
from __future__ import annotations

import re
import statistics
import time

from .generation import generate
from .retrieval import HybridRetriever
from .stores import Document, InMemoryDocumentStore, InMemoryGraphStore

GOLDEN_DOCS = [
    Document(id="refund-policy", text="Refunds are issued within 14 days of purchase with a receipt."),
    Document(id="shipping-policy", text="Standard shipping takes 3 to 5 business days and is free over $50."),
    Document(id="support-hours", text="Phone support is available Monday to Friday, 9 AM to 6 PM Eastern."),
]

# (question, ids that MUST be cited)
GOLDEN_QUESTIONS = [
    ("How long do refunds take?", {"refund-policy"}),
    ("How fast is standard shipping?", {"shipping-policy"}),
    ("When can I call support?", {"support-hours"}),
    ("What is your Martian exchange policy?", set()),  # unanswerable: must abstain
]


def _answer_text(out: dict) -> str:
    return out["answer"]


def faithfulness(out: dict, required_ids: set[str]) -> bool:
    """Every required doc id must appear as a citation."""
    text = _answer_text(out)
    cited = set(re.findall(r"\[doc:([^\]]+)\]", text)) | set(out.get("citations", []))
    return required_ids <= cited


def abstention(out: dict) -> bool:
    text = _answer_text(out).lower()
    return "don't have" in text or "does not contain" in text or "no ingested" in text


def run() -> dict:
    docs, graph = InMemoryDocumentStore(), InMemoryGraphStore()
    retriever = HybridRetriever(docs, graph)
    retriever.ingest(GOLDEN_DOCS)

    latencies: list[float] = []
    faithful = 0
    abstained = 0
    for question, required in GOLDEN_QUESTIONS:
        start = time.perf_counter()
        results = retriever.retrieve(question)
        out = generate(question, results)
        latencies.append((time.perf_counter() - start) * 1000)
        if not required:
            abstained += abstention(out)
        elif faithfulness(out, required):
            faithful += 1
        else:
            print(f"FAIL faithfulness: {question}\n  -> {_answer_text(out)[:200]}")

    answerable = sum(1 for _, r in GOLDEN_QUESTIONS if r)
    report = {
        "faithfulness": f"{faithful}/{answerable}",
        "abstention_correct": f"{abstained}/1",
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1)
            if len(latencies) > 1 else round(latencies[0], 1),
        },
    }
    print(report)
    ok = faithful == answerable and abstained == 1
    print("EVALS PASSED" if ok else "EVALS FAILED")
    return report


if __name__ == "__main__":
    run()
