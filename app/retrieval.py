"""Hybrid retriever: vector search -> graph expansion -> rerank.

The pipeline is deliberately explicit so a client can see (and audit) every
stage: what was retrieved, what the graph added, and why the final set won.
"""
from __future__ import annotations

from dataclasses import dataclass

from .stores import Document


@dataclass
class RetrievalResult:
    document: Document
    vector_score: float
    graph_boost: float = 0.0
    final_score: float = 0.0
    via_graph: bool = False


class HybridRetriever:
    def __init__(self, doc_store, graph_store, vector_k: int = 8, final_k: int = 4,
                 min_score: float = 0.05):
        self.doc_store = doc_store
        self.graph_store = graph_store
        self.vector_k = vector_k
        self.final_k = final_k
        # Below this score a hit is noise, not evidence. Returning nothing
        # lets the generator abstain instead of hallucinating.
        self.min_score = min_score

    def ingest(self, documents: list[Document]) -> int:
        self.doc_store.add(documents)
        self.graph_store.add(documents)
        return len(documents)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        vector_hits = self.doc_store.search(query, k=self.vector_k)
        results = [
            RetrievalResult(document=d, vector_score=s, final_score=s)
            for d, s in vector_hits
        ]
        seed_ids = [r.document.id for r in results]

        # Graph expansion: pull docs related to the seed set through entities.
        expanded_ids = self.graph_store.expand(seed_ids)
        by_id = {r.document.id: r for r in results}
        for doc_id in expanded_ids:
            if doc_id not in by_id:
                # Fetch the doc object via a targeted search on its id.
                hits = self.doc_store.search(doc_id, k=1)
                if hits:
                    d, _ = hits[0]
                    results.append(
                        RetrievalResult(
                            document=d, vector_score=0.0,
                            graph_boost=0.25, final_score=0.25, via_graph=True,
                        )
                    )

        # Rerank: vector score wins, graph expansion gets a fixed boost.
        # Drop anything below the noise floor so the generator can abstain.
        for r in results:
            r.final_score = r.vector_score + r.graph_boost
        results.sort(key=lambda r: r.final_score, reverse=True)
        return [r for r in results[: self.final_k] if r.final_score >= self.min_score]

    def explain(self, query: str) -> dict:
        """Auditable trace of a retrieval call — show this to the client."""
        results = self.retrieve(query)
        return {
            "query": query,
            "stages": ["vector_search", "graph_expansion", "rerank"],
            "results": [
                {
                    "id": r.document.id,
                    "vector_score": round(r.vector_score, 4),
                    "graph_boost": r.graph_boost,
                    "final_score": round(r.final_score, 4),
                    "via_graph": r.via_graph,
                    "snippet": r.document.text[:160],
                }
                for r in results
            ],
        }
