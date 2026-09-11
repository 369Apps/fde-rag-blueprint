"""Answer generation: cited answers in demo mode, OpenAI in production.

Every answer carries citations back to document ids. A RAG answer without
citations is a liability; this module makes them structural, not optional.
"""
from __future__ import annotations

import os

from .retrieval import RetrievalResult

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "Cite every factual claim as [doc:<id>]. If the context does not "
    "contain the answer, say so explicitly instead of guessing."
)


def _demo_answer(question: str, results: list[RetrievalResult]) -> dict:
    if not results:
        return {
            "answer": "I don't have any ingested documents that cover this.",
            "citations": [],
            "mode": "demo",
        }
    parts = []
    for r in results:
        parts.append(f"[doc:{r.document.id}] {r.document.text.strip()}")
    return {
        "answer": (
            "Based on the retrieved context:\n\n" + "\n\n".join(parts)
        ),
        "citations": [r.document.id for r in results],
        "mode": "demo",
    }


def _openai_answer(question: str, results: list[RetrievalResult]) -> dict:
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    context = "\n\n".join(
        f"[doc:{r.document.id}] {r.document.text}" for r in results
    )
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
    )
    return {
        "answer": resp.choices[0].message.content,
        "citations": [r.document.id for r in results],
        "mode": "openai",
    }


def generate(question: str, results: list[RetrievalResult]) -> dict:
    if os.environ.get("OPENAI_API_KEY"):
        return _openai_answer(question, results)
    return _demo_answer(question, results)
