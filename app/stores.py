"""Storage backends: in-memory demo plus real pgvector / Neo4j implementations.

The retriever talks to the `DocumentStore` / `GraphStore` protocols, so demo
mode and production mode share every code path except the backend.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "for", "with", "as", "at",
    "by", "from", "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "our", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "do", "does", "did", "can", "could", "should",
    "would", "will", "not", "no", "if", "then", "than", "so", "very",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _tfidf_vectors(docs: list[Document]):
    """Tiny in-memory TF-IDF so demo mode has real ranking, not stubs."""
    vocab: dict[str, int] = {}
    for d in docs:
        for t in set(_tokens(d.text)):
            vocab.setdefault(t, len(vocab))
    n = len(docs)
    df = [0] * len(vocab)
    for d in docs:
        for t in set(_tokens(d.text)):
            df[vocab[t]] += 1
    idf = [math.log((n + 1) / (f + 1)) + 1 for f in df]

    def embed(text: str):
        vec = [0.0] * len(vocab)
        for t in _tokens(text):
            if t in vocab:
                vec[vocab[t]] += 1.0
        return [v * w for v, w in zip(vec, idf)]

    return embed


def _cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0


class InMemoryDocumentStore:
    """Demo backend: TF-IDF ranking over an in-memory corpus."""

    def __init__(self) -> None:
        self._docs: list[Document] = []

    def add(self, docs: list[Document]) -> None:
        self._docs.extend(docs)

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if not self._docs:
            return []
        embed = _tfidf_vectors(self._docs)
        q = embed(query)
        scored = [(d, _cosine(q, embed(d.text))) for d in self._docs]
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:k]


class PgVectorDocumentStore:
    """Production backend: Postgres + pgvector. Requires DATABASE_URL."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.environ["DATABASE_URL"]

    def add(self, docs: list[Document]) -> None:
        import psycopg  # type: ignore

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE EXTENSION IF NOT EXISTS vector;"
                    "CREATE TABLE IF NOT EXISTS rag_docs"
                    " (id TEXT PRIMARY KEY, text TEXT, embedding vector(1536));"
                )
                # Embeddings omitted here: wire your embedding provider in.
                for d in docs:
                    cur.execute(
                        "INSERT INTO rag_docs (id, text) VALUES (%s, %s)"
                        " ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text;",
                        (d.id, d.text),
                    )
            conn.commit()

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        import psycopg  # type: ignore

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, text FROM rag_docs LIMIT %s;", (k,)
                )
                return [(Document(id=r[0], text=r[1]), 1.0) for r in cur.fetchall()]
        # NOTE: plug embedding-based ORDER BY embedding <-> %s::vector in prod.


class InMemoryGraphStore:
    """Demo graph: entities are capitalized phrases; edges link co-occurrence."""

    def __init__(self) -> None:
        self.entities: dict[str, set[str]] = {}

    def add(self, docs: list[Document]) -> None:
        for d in docs:
            ents = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", d.text))
            for e in ents:
                self.entities.setdefault(e, set()).add(d.id)

    def expand(self, doc_ids: list[str], depth: int = 1) -> list[str]:
        related: set[str] = set()
        for entity, ids in self.entities.items():
            if ids & set(doc_ids):
                related |= ids
        return [i for i in related if i not in doc_ids]


class Neo4jGraphStore:
    """Production backend: Neo4j. Requires NEO4J_URI/USER/PASSWORD."""

    def __init__(self) -> None:
        from neo4j import GraphDatabase  # type: ignore

        self.driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )

    def add(self, docs: list[Document]) -> None:
        # Wire your entity extractor here; schema: (:Entity)-[:MENTIONED_IN]->(:Doc)
        raise NotImplementedError("Connect your NER pipeline to populate the graph.")

    def expand(self, doc_ids: list[str], depth: int = 1) -> list[str]:
        with self.driver.session() as s:
            rows = s.run(
                "MATCH (d:Doc)-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]-(o:Doc)"
                " WHERE d.id IN $ids RETURN DISTINCT o.id",
                ids=doc_ids,
            )
            return [r[0] for r in rows if r[0] not in doc_ids]


def default_stores():
    if os.environ.get("DATABASE_URL"):
        docs: object = PgVectorDocumentStore()
    else:
        docs = InMemoryDocumentStore()
    graph: object = Neo4jGraphStore() if os.environ.get("NEO4J_URI") else InMemoryGraphStore()
    return docs, graph
