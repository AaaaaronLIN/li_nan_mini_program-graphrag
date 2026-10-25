# Mini GraphRAG Fraud  Results (k=5)

- Documents: 900
- Queries: 30
- Claims: 300

| Method | Query type | n | Recall@k | Precision@k | MRR@k |
|---|---|---:|---:|---:|---:|
| GraphRAG | billing_review_suspicion | 4 | 0.9167 | 0.5000 | 0.8750 |
| GraphRAG | global_top_provider | 2 | 0.8333 | 1.0000 | 1.0000 |
| GraphRAG | provider_high_amount | 8 | 0.8333 | 1.0000 | 1.0000 |
| GraphRAG | shared_patient_multihop | 8 | 0.8333 | 1.0000 | 1.0000 |
| GraphRAG | single_claim_support | 8 | 1.0000 | 0.6000 | 1.0000 |
| HybridRAG | billing_review_suspicion | 4 | 0.9167 | 0.5000 | 0.8750 |
| HybridRAG | global_top_provider | 2 | 0.6667 | 0.8000 | 1.0000 |
| HybridRAG | provider_high_amount | 8 | 0.6667 | 0.8000 | 1.0000 |
| HybridRAG | shared_patient_multihop | 8 | 0.6667 | 0.8000 | 1.0000 |
| HybridRAG | single_claim_support | 8 | 1.0000 | 0.6000 | 1.0000 |
| TextRAG | billing_review_suspicion | 4 | 0.9167 | 0.5000 | 0.8750 |
| TextRAG | global_top_provider | 2 | 0.0000 | 0.0000 | 0.0000 |
| TextRAG | provider_high_amount | 8 | 0.1667 | 0.2000 | 0.3229 |
| TextRAG | shared_patient_multihop | 8 | 0.2292 | 0.2750 | 0.4417 |
| TextRAG | single_claim_support | 8 | 0.6667 | 0.4000 | 1.0000 |

> Retrieval-only benchmark on synthetic data. No LLM generation or human evaluation yet.
