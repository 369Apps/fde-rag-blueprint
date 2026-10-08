"""Empty retrieval must abstain before any model call."""
import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from app.generation import generate
from app.retrieval import RetrievalResult
from app.stores import Document


def _fake_openai(constructed: list, on_create=None):
    class _Completions:
        def create(self, **kwargs):
            if on_create is not None:
                on_create(kwargs)
            raise AssertionError("chat.completions.create was called")

    class _Chat:
        completions = _Completions()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            constructed.append(True)
            self.chat = _Chat()

    fake = ModuleType("openai")
    fake.OpenAI = FakeOpenAI
    return fake


class EmptyRetrievalAbstainTests(unittest.TestCase):
    def test_empty_retrieval_abstains_without_calling_the_model(self):
        constructed: list[bool] = []
        fake = _fake_openai(constructed)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch.dict(sys.modules, {"openai": fake}):
                out = generate("What is your Martian exchange policy?", [])
        self.assertEqual(constructed, [])
        self.assertIs(out["abstained"], True)
        self.assertIn("don't have", out["answer"])
        self.assertEqual(out["citations"], [])
        self.assertEqual(out["mode"], "openai")

    def test_empty_retrieval_abstains_in_demo_mode(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            out = generate("What is your Martian exchange policy?", [])
        self.assertIs(out["abstained"], True)
        self.assertIn("don't have", out["answer"])
        self.assertEqual(out["citations"], [])
        self.assertEqual(out["mode"], "demo")

    def test_nonempty_retrieval_still_reaches_the_model(self):
        constructed: list[bool] = []
        created: list[dict] = []
        fake = _fake_openai(constructed, on_create=created.append)
        results = [
            RetrievalResult(
                document=Document(id="refund-policy", text="Refunds take 14 days."),
                vector_score=1.0,
            )
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch.dict(sys.modules, {"openai": fake}):
                with self.assertRaises(AssertionError):
                    generate("How long do refunds take?", results)
        self.assertEqual(constructed, [True])
        self.assertEqual(len(created), 1)
        self.assertIn("Question: How long do refunds take?", created[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
