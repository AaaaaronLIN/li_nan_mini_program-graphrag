# One-page Research Brief

## Working title
**Does Graph Structure Help LLM Fraud Investigation? A Reproducible GraphRAG Benchmark for Medical Insurance Claims**

## Problem
Fraud investigation needs multi-hop evidence chains across patients, providers, claims, payments and documents. Standard text RAG retrieves document chunks by similarity and can miss the relational chain. GraphRAG may improve evidence recall and explanation quality, but reproducible fraud-specific benchmarks are lacking.

## Research questions
- RQ1: Does GraphRAG improve multi-hop evidence retrieval over text RAG?
- RQ2: Does graph-structured evidence improve fraud classification and explanation faithfulness?
- RQ3: What are the latency and token-cost trade-offs?

## Data and schema
Synthetic data first; replace with anonymised project data under a data-use agreement.
Entities: Patient, Provider, Claim, Document, Payment.
Relations: treated_by, billed, has_patient, supported_by, pays.
Schema: `data/schema.json`.

## Methods
Planned baseline set; the current prototype implements 3, 5 and 6 only.
1. LLM-only, no retrieval (planned)
2. BM25 text RAG (planned; prototype uses TF-IDF)
3. TF-IDF text RAG (implemented)
4. Dense text RAG (planned)
5. GraphRAG: entity/relation extraction -> graph -> 1-2 hop subgraph retrieval (implemented, rule-based operators)
6. Hybrid RAG (implemented, naive rank fusion)

## Evaluation
- Retrieval: Recall@k, Precision@k, MRR@k
- Answer: accuracy / macro-F1
- Explanation: citation precision and faithfulness
- Cost: latency, token count, cost per query
- Optional human study: randomised text RAG vs GraphRAG; outcomes = fraud detection accuracy, false-positive rate, confidence calibration (Brier score), trust rating, decision time

## Deliverables
- Reproducible GitHub repository
- 30-50 query benchmark with gold evidence
- Baseline results and error analysis
- 6-8 page technical report
- Short demo

## Four-week plan
Week 1: schema, synthetic data, 30 queries, gold evidence.
Week 2: text RAG and GraphRAG baselines; evaluation harness.
Week 3: experiments, error analysis, citation/evidence-chain evaluation.
Week 4: demo, report, and a one-page extension proposal for a randomised human study.

## My proposed contribution
I can own the schema, benchmark construction, baseline implementation, evaluation metrics,
statistical analysis, and technical report. I would like to discuss whether this mini
benchmark could be useful to the GraphRAG fraud project, whether anonymised data could
be used, and whether the work could become an FYP / MPhil application project.
