"""Regression tests for eval scoring. These lock the three harness bugs."""
import unittest

from app.evals import GOLDEN_QUESTIONS, abstention, build_report, faithfulness, p95, run


class FaithfulnessTests(unittest.TestCase):
    def test_retrieved_id_without_text_citation_is_not_faithful(self):
        out = {"answer": "Refunds take 14 days.", "citations": ["refund-policy"]}
        self.assertFalse(faithfulness(out, {"refund-policy"}))

    def test_text_citation_is_faithful_even_if_citations_field_is_empty(self):
        out = {"answer": "Refunds take 14 days [doc:refund-policy].", "citations": []}
        self.assertTrue(faithfulness(out, {"refund-policy"}))

    def test_citations_field_cannot_supply_a_missing_text_citation(self):
        out = {
            "answer": "See [doc:shipping-policy].",
            "citations": ["refund-policy", "shipping-policy"],
        }
        self.assertFalse(faithfulness(out, {"refund-policy"}))


class AbstentionTests(unittest.TestCase):
    def test_structural_flag_true(self):
        self.assertTrue(abstention({"answer": "Sure, here is a guess.", "abstained": True}))

    def test_structural_flag_overrides_abstention_phrase(self):
        out = {
            "answer": "I don't have a 30 day window; refunds take 14 days [doc:refund-policy].",
            "abstained": False,
        }
        self.assertFalse(abstention(out))

    def test_phrase_fallback_when_flag_absent(self):
        self.assertTrue(
            abstention({"answer": "I don't have any ingested documents that cover this."})
        )
        self.assertFalse(abstention({"answer": "Refunds take 14 days [doc:refund-policy]."}))


class P95Tests(unittest.TestCase):
    def test_four_samples_is_the_max_not_the_third(self):
        # int(4 * 0.95) - 1 == 2, which selected 30.
        self.assertEqual(p95([10, 20, 30, 100]), 100.0)

    def test_twenty_samples_is_rank_19(self):
        self.assertEqual(p95([float(i) for i in range(1, 21)]), 19.0)

    def test_twenty_one_samples_is_rank_20(self):
        # int(21 * 0.95) - 1 == 18, which selected 19.
        self.assertEqual(p95([float(i) for i in range(1, 22)]), 20.0)

    def test_single_sample(self):
        self.assertEqual(p95([4.26]), 4.3)


class ReportTests(unittest.TestCase):
    def test_abstention_denominator_follows_the_question_set(self):
        questions = [("a", {"doc"}), ("b", set()), ("c", set())]
        report, ok = build_report(1, 2, questions, [10, 20, 30, 40])
        self.assertEqual(report["faithfulness"], "1/1")
        self.assertEqual(report["abstention_correct"], "2/2")
        self.assertTrue(ok)

        report, ok = build_report(1, 1, questions, [10, 20, 30, 40])
        self.assertEqual(report["abstention_correct"], "1/2")
        self.assertFalse(ok)

    def test_small_sample_p95_note(self):
        report, _ = build_report(1, 0, [("a", {"doc"})], [5, 15])
        self.assertEqual(report["latency_ms"]["samples"], 2)
        self.assertEqual(report["latency_ms"]["p95"], 15.0)
        self.assertEqual(report["latency_ms"]["note"], "p95 is the max below 20 samples")

    def test_note_blank_at_20_samples(self):
        report, _ = build_report(0, 0, [], [float(i) for i in range(1, 21)])
        self.assertEqual(report["latency_ms"]["p95"], 19.0)
        self.assertEqual(report["latency_ms"]["samples"], 20)
        self.assertEqual(report["latency_ms"]["note"], "")

    def test_golden_run_passes(self):
        report = run()
        unanswerable = sum(1 for _, required in GOLDEN_QUESTIONS if not required)
        answerable = len(GOLDEN_QUESTIONS) - unanswerable
        self.assertEqual(report["faithfulness"], f"{answerable}/{answerable}")
        self.assertEqual(report["abstention_correct"], f"{unanswerable}/{unanswerable}")
        self.assertEqual(report["latency_ms"]["samples"], len(GOLDEN_QUESTIONS))
        self.assertIn("max", report["latency_ms"]["note"])


if __name__ == "__main__":
    unittest.main()
