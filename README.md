# Mini GraphRAG Fraud Benchmark

A small, dependency-free (pure Python standard library) retrieval benchmark for a
medical-insurance fraud investigation task. It asks one question: **does graph structure
help retrieve multi-hop evidence, compared with plain text retrieval?**

Built as a pre-contact prototype for Prof. Nan Li's project *"GraphRAG Empowered
Investigation System for Fraud, Waste, Abuse in Medical Insurance Claim"*.

## One-command run

```bash
cd li_nan_mini_program
python3 main.py --k 5
```

No `pip install` is required. The program uses only the Python standard library.

Expected output:

```text
Mini GraphRAG Fraud Benchmark
documents=900 queries=30

method     query_type                      n   recall  precision      mrr
GraphRAG   provider_high_amount           8    0.833      1.000    1.000
TextRAG    provider_high_amount           8    0.167      0.200    0.323
...
```

## What it does

1. Generates 30 providers, 120 patients, 300 claims, 900 documents, and 30 queries (seed 42).
2. Builds a directed knowledge graph:
   - Patient --treated_by--> Provider
   - Provider --billed--> Claim
   - Claim --has_patient--> Patient
   - Claim --supported_by--> Document
   - Payment --pays--> Claim
3. Runs three retrievers:
   - `TextRAG`: pure-Python TF-IDF retrieval over documents.
   - `GraphRAG`: rule-based graph operators + graph expansion, then document ranking.
   - `HybridRAG`: naive rank fusion of text and graph results.
4. Evaluates Recall@k, Precision@k, and MRR@k per query type.
5. Writes `outputs/results.csv`, `outputs/results.md`, and `outputs/summary.json`.

## Results (k=5, synthetic data)

| Query type | n | GraphRAG Recall@5 | TextRAG Recall@5 |
|---|---:|---:|---:|
| single_claim_support | 8 | 1.000 | 0.667 |
| provider_high_amount | 8 | 0.833 | 0.167 |
| shared_patient_multihop | 8 | 0.833 | 0.229 |
| billing_review_suspicion | 4 | 0.917 | 0.917 |
| global_top_provider | 2 | 0.833 | 0.000 |

Two things to read carefully before quoting these numbers:

- **0.833 is the ceiling, not a near-miss.** Those queries have 6 gold documents, so at
  k=5 the maximum achievable recall is 5/6 = 0.833. GraphRAG is at the ceiling.
- **The text baseline is deliberately simple** (TF-IDF, no numeric filtering, no BM25,
  and it is not given the seed entity IDs). The gap is therefore an upper bound on the
  benefit of graph structure, not a fair estimate of it.

## Code map

| File | Purpose |
|---|---|
| `main.py` | one-command entry point |
| `mini_graphrag/dataset.py` | synthetic insurance data + benchmark queries |
| `mini_graphrag/graph.py` | tiny directed graph implementation |
| `mini_graphrag/text_rag.py` | TF-IDF text retrieval |
| `mini_graphrag/graph_rag.py` | GraphRAG and HybridRAG retrieval |
| `mini_graphrag/evaluate.py` | metrics and report writers |
| `data/schema.json` | entity/relation schema |
| `tests/test_smoke.py` | smoke tests |
| `research_brief.md` | one-page research plan for Prof. Li |

## Demo one query

```bash
python3 main.py --demo "Which documents support the highest-amount claims billed by provider P001?" --k 5
```

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Limitations

- Synthetic data only, not real insurance claims; the generator is idealised (no missing
  fields, no duplicate or misspelled entity IDs).
- **Gold evidence comes from the same generator as the graph**, so GraphRAG partly
  recovers the structure that produced the labels. The comparison is not independent.
- Retrieval-only; no LLM answer generation, no citation faithfulness, no answer accuracy.
- GraphRAG uses query-type-specific graph operators (string-matched branches), not a
  general LLM graph planner.
- Small benchmark (30 queries, providers P001-P008 only); no confidence intervals and no
  human evaluation yet.
- Known bug found during error analysis: medical-note documents do not contain the claim
  ID in the body text, so any claim-ID query can never retrieve them. This explains why
  TextRAG is at 0.667 on the easiest query type.

## Next research step

Add an LLM generator with citation grounding, make gold labels independent of the graph,
replace rule-based graph operators with a graph query planner, and optionally run a
randomised human study comparing text RAG and GraphRAG on decision accuracy, false
positives, confidence calibration, and trust.
