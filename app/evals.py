"""Eval harness: golden question set + faithfulness / abstention / latency.

Run:  python -m app.evals
Gate deployments on this. A RAG pipeline that can't pass its own evals
should not be taking customer questions.
"""
from __future__ import annotations

import math
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

# (question, ids that MUST be cited). An empty set is unanswerable: abstain.
GOLDEN_QUESTIONS = [
    ("How long do refunds take?", {"refund-policy"}),
    ("How fast is standard shipping?", {"shipping-policy"}),
    ("When can I call support?", {"support-hours"}),
    ("What is your Martian exchange policy?", set()),
]


def _answer_text(out: dict) -> str:
    return out["answer"]


def faithfulness(out: dict, required_ids: set[str]) -> bool:
    """Every required doc id must be cited in the answer text.

    The citations field lists retrieved ids, including ones the answer never
    mentions. Counting those measured retrieval, not faithfulness.
    """
    cited = set(re.findall(r"\[doc:([^\]]+)\]", _answer_text(out)))
    return required_ids <= cited


def abstention(out: dict) -> bool:
    """Structural flag first; phrase match only as a fallback.

    The flag wins when both are present, so an answer that says "don't have"
    but did not abstain is not scored as an abstention.
    """
    if "abstained" in out:
        return bool(out["abstained"])
    text = _answer_text(out).lower()
    return "don't have" in text or "does not contain" in text or "no ingested" in text


def p95(latencies: list[float]) -> float:
    """Nearest-rank p95: value at rank ceil(0.95 * n).

    With fewer than 20 samples that rank is n, so this is the maximum.
    Rank is computed in integer arithmetic; float ceil(0.95 * n) can land
    just below an integer and select the next-lower sample.
    """
    ordered = sorted(latencies)
    n = len(ordered)
    rank = max(1, (95 * n + 99) // 100)
    return round(ordered[rank - 1], 1)


def build_report(
    faithful: int,
    abstained: int,
    questions: list[tuple[str, set[str]]],
    latencies: list[float],
) -> tuple[dict, bool]:
    answerable = sum(1 for _, required in questions if required)
    unanswerable = len(questions) - answerable
    report = {
        "faithfulness": f"{faithful}/{answerable}",
        "abstention_correct": f"{abstained}/{unanswerable}",
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p95": p95(latencies),
            "samples": len(latencies),
            "note": "p95 is the max below 20 samples" if len(latencies) < 20 else "",
        },
    }
    ok = faithful == answerable and abstained == unanswerable
    return report, ok


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

    report, ok = build_report(faithful, abstained, GOLDEN_QUESTIONS, latencies)
    print(report)
    print("EVALS PASSED" if ok else "EVALS FAILED")
    return report


if __name__ == "__main__":
    run()
