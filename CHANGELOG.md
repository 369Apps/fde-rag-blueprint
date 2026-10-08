# Changelog

## Eval harness scoring

Three bugs in `app/evals.py` made the golden-set report agree with retrieval and a hardcoded denominator instead of with the answer.

- **Faithfulness** counted ids in the `citations` field, which the generator fills with every retrieved document. An answer that never wrote `[doc:<id>]` still passed. Measured counterexample: `{"answer": "Refunds take 14 days.", "citations": ["refund-policy"]}` scored faithful before this fix.
- **Abstention** printed `abstained/1` and passed only when that count was 1. The golden set has one unanswerable question, so the printed ratio was `1/1` either way. Two correct abstentions used to report `2/1` and fail the gate. Phrase matching is now only the fallback; an `abstained` flag on the answer wins.
- **p95** used `sorted(latencies)[int(n * 0.95) - 1]`. With the golden set's 4 samples that index is 2 (the 3rd value). Nearest-rank p95 is `ceil(0.95 * n)`, which is the maximum below 20 samples. The report says so.

### Measured before

`python -m app.evals` on the unfixed harness:

```
{'faithfulness': '3/3', 'abstention_correct': '1/1', 'latency_ms': {'p50': 0.1, 'p95': 0.1}}
EVALS PASSED
```

On one golden run the sorted latencies were 0.044, 0.0521, 0.062, and 0.1206 ms. The old index selected 0.062 ms (reported 0.1) and nearest-rank selected 0.1206 ms (reported 0.1). Across 30 repeats of the same four questions, the rounded p95 differed once: sorted samples 0.0811, 0.0914, 0.1013, 0.1896 ms reported 0.1 under the old formula and 0.2 under nearest-rank.

Golden-set faithfulness stayed 3/3 because demo answers already contain `[doc:<id>]`. Abstention stayed 1/1 because there is one unanswerable question and the demo generator abstains on it.

### Measured after

`python -m app.evals` on this fix:

```
{'faithfulness': '3/3', 'abstention_correct': '1/1', 'latency_ms': {'p50': 0.1, 'p95': 0.1, 'samples': 4, 'note': 'p95 is the max below 20 samples'}}
EVALS PASSED
```

`python -m unittest discover -s tests -v` ran 14 tests, all passing. On this run the 1-decimal p50 and p95 stayed 0.1; the report now includes the sample count and the small-sample note. The 0.1 vs 0.2 gap above is the same timings scored both ways, not a second harness invocation.

Deterministic checks of the same formulas, not wall-clock timings: latencies 10, 20, 30, 40 ms scored p95 **30.0** before and **40.0** after. Two correct abstentions on a three-question set scored **2/1** and failed the old gate, and scored **2/2** and passed after.

## Empty retrieval abstains before the model

`generate` called `_openai_answer` whenever `OPENAI_API_KEY` was set, including when retrieval returned no documents. That sent the model an empty context. Confirmed by substituting a fake OpenAI client: `generate(question, [])` with a key set constructed the client.

Demo mode already returned "I don't have any ingested documents that cover this." for an empty result list, but it did not set `abstained`, and that return sat behind the API-key branch.

Empty results now return that same sentence with `abstained=True` and `citations=[]` before either generator runs. With a key set, `mode` is `openai`; otherwise `demo`. A non-empty result list still reaches the model.
