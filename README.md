# Li Nan Mini  Mini GraphRAG Fraud Benchmark

A complete, dependency-free mini program created before contacting Prof. Li Nan.
It tests whether graph structure helps retrieve multi-hop evidence in a synthetic
medical-insurance fraud investigation task.

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

GraphRAG   provider_high_amount     n=8 recall@5=0.833 ...
TextRAG    provider_high_amount     n=8 recall@5=0.167 ...
```

## What it does

1. Generates 30 providers, 120 patients, 300 claims, 900 documents, and 30 queries.
2. Builds a directed knowledge graph:
   - Patient --treated_by--> Provider
   - Provider --billed--> Claim
   - Claim --has_patient--> Patient
   - Claim --supported_by--> Document
   - Payment --pays--> Claim
3. Runs three retrievers:
   - `TextRAG`: pure-Python TF-IDF retrieval over documents.
   - `GraphRAG`: structured graph queries + graph expansion, then document ranking.
   - `HybridRAG`: naive fusion of text and graph results.
4. Evaluates Recall@k, Precision@k, and MRR@k.
5. Writes `outputs/results.csv`, `outputs/results.md`, and `outputs/summary.json`.

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
| `email_li_nan.md` | email draft |

## Demo one query

```bash
python3 main.py --demo "Which documents support the highest-amount claims billed by provider P001?" --k 5
```

## Run tests

```bash
python3 -m unittest discover -s tests -v
```

## Limitations

- Synthetic data only, not real insurance claims.
- Retrieval-only; no LLM answer generation yet.
- GraphRAG uses query-type-specific graph operators, not a general LLM graph planner.
- Small benchmark (30 queries); no confidence intervals or human evaluation yet.

## Next research step

Add an LLM generator with citation grounding, replace rule-based graph operators with
a graph query planner, and optionally run a randomised human study comparing text RAG
and GraphRAG on decision accuracy, false positives, confidence calibration, and trust.
