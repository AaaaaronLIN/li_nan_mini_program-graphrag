"""Synthetic medical-insurance claims and a small fraud-investigation benchmark.

The dataset is intentionally synthetic. It is designed to test retrieval over
multi-hop relations, not to make claims about real fraud rates.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _amount(rng: random.Random, fraud: bool) -> float:
    base = rng.lognormvariate(8.7, 0.65)
    factor = rng.uniform(1.8, 4.0) if fraud else rng.uniform(0.7, 1.2)
    return round(base * factor, 2)


def generate_dataset(seed: int = 42, n_providers: int = 30,
                     n_patients: int = 120, n_claims: int = 300) -> dict[str, Any]:
    rng = random.Random(seed)
    specialties = ["cardiology", "oncology", "orthopedics", "dermatology", "internal medicine"]
    regions = ["NT", "KLN", "HKI"]

    providers = []
    for i in range(1, n_providers + 1):
        fraud_group = i <= 6
        providers.append({
            "provider_id": f"P{i:03d}",
            "specialty": rng.choice(specialties),
            "region": rng.choice(regions),
            "risk_score": round(rng.uniform(0.65, 0.95) if fraud_group else rng.uniform(0.02, 0.35), 3),
            "fraud_group": fraud_group,
        })

    patients = []
    for i in range(1, n_patients + 1):
        patients.append({
            "patient_id": f"S{i:03d}",
            "age": rng.randint(18, 85),
            "sex": rng.choice(["F", "M"]),
            "region": rng.choice(regions),
        })

    claims = []
    for i in range(1, n_claims + 1):
        provider = rng.choice(providers)
        patient = rng.choice(patients)
        is_fraud = bool(provider["fraud_group"] and rng.random() < 0.55)
        claims.append({
            "claim_id": f"C{i:04d}",
            "patient_id": patient["patient_id"],
            "provider_id": provider["provider_id"],
            "service_date": f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
            "amount": _amount(rng, is_fraud),
            "diagnosis": rng.choice(["I10", "E11", "M54", "J45", "C50", "K21"]),
            "procedure": rng.choice(["99213", "99214", "99215", "93000", "71046", "27447"]),
            "status": "flagged" if is_fraud else "paid",
            "is_fraud": is_fraud,
        })

    documents = []
    for c in claims:
        cid, pid, sid = c["claim_id"], c["provider_id"], c["patient_id"]
        documents.append({
            "doc_id": f"D{cid}_summary",
            "claim_id": cid,
            "doc_type": "claim_summary",
            "text": (
                f"Claim {cid} was billed by provider {pid} for patient {sid}. "
                f"Amount HK${c['amount']:.2f}. Diagnosis {c['diagnosis']}, procedure {c['procedure']}. "
                f"Status {c['status']}."
            ),
        })
        documents.append({
            "doc_id": f"D{cid}_note",
            "claim_id": cid,
            "doc_type": "medical_note",
            "text": (
                f"Medical note for patient {sid} treated by {pid}. "
                f"Diagnosis {c['diagnosis']}; procedure {c['procedure']}. "
                f"Documented service may not fully justify the billed amount HK${c['amount']:.2f}."
            ),
        })
        suspicious = "High amount, possible upcoding, duplicate pattern, or kickback relationship." if c["is_fraud"] else "No major billing anomaly detected."
        documents.append({
            "doc_id": f"D{cid}_review",
            "claim_id": cid,
            "doc_type": "billing_review",
            "text": (
                f"Billing review for claim {cid} from provider {pid}. "
                f"Amount HK${c['amount']:.2f}. {suspicious} Patient {sid}."
            ),
        })

    payments = [{
        "payment_id": f"PAY{i:04d}",
        "claim_id": c["claim_id"],
        "amount": c["amount"],
        "date": c["service_date"],
    } for i, c in enumerate(claims, 1)]

    queries: list[dict[str, Any]] = []
    def add_query(qtype: str, question: str, seeds: list[str], gold: list[str]) -> None:
        queries.append({
            "query_id": f"Q{len(queries) + 1:03d}",
            "query_type": qtype,
            "question": question,
            "seed_entities": seeds,
            "gold_doc_ids": gold,
        })

    def docs_for_claim(cid: str) -> list[str]:
        return [f"D{cid}_summary", f"D{cid}_note", f"D{cid}_review"]

    # 8 single-hop claim-support questions.
    for c in claims[:8]:
        add_query(
            "single_claim_support",
            f"Which documents support claim {c['claim_id']}?",
            [c["claim_id"]],
            docs_for_claim(c["claim_id"]),
        )

    # 8 provider high-amount questions.
    for p in providers[:8]:
        pclaims = sorted(
            [c for c in claims if c["provider_id"] == p["provider_id"]],
            key=lambda x: x["amount"], reverse=True,
        )[:2]
        gold = [doc for c in pclaims for doc in docs_for_claim(c["claim_id"])]
        add_query(
            "provider_high_amount",
            f"Which documents support the highest-amount claims billed by provider {p['provider_id']}?",
            [p["provider_id"]],
            gold,
        )

    # 8 shared-patient multi-hop questions.
    for p in providers[:8]:
        pclaims = [c for c in claims if c["provider_id"] == p["provider_id"]]
        shared_claims = []
        for c in pclaims:
            sid = c["patient_id"]
            other = [x for x in claims if x["patient_id"] == sid and x["provider_id"] != p["provider_id"]]
            if other:
                shared_claims.append(c)
        shared_claims = sorted(shared_claims, key=lambda x: x["amount"], reverse=True)[:2]
        gold = [doc for c in shared_claims for doc in docs_for_claim(c["claim_id"])]
        add_query(
            "shared_patient_multihop",
            f"Which documents support suspicious claims from provider {p['provider_id']} whose patients also appear under other providers?",
            [p["provider_id"]],
            gold,
        )

    # 4 billing-review questions.
    for p in providers[:4]:
        review_claims = [c for c in claims if c["provider_id"] == p["provider_id"] and c["is_fraud"]][:3]
        gold = [f"D{c['claim_id']}_review" for c in review_claims]
        add_query(
            "billing_review_suspicion",
            f"Which billing review documents for provider {p['provider_id']} mention possible upcoding or kickback?",
            [p["provider_id"]],
            gold,
        )

    # 2 global questions.
    avg = {}
    for p in providers:
        pclaims = [c for c in claims if c["provider_id"] == p["provider_id"]]
        avg[p["provider_id"]] = sum(c["amount"] for c in pclaims) / max(1, len(pclaims))
    top_pid = max(avg, key=avg.get)
    top_claims = sorted([c for c in claims if c["provider_id"] == top_pid], key=lambda x: x["amount"], reverse=True)[:2]
    top_gold = [doc for c in top_claims for doc in docs_for_claim(c["claim_id"])]
    add_query(
        "global_top_provider",
        "Which provider has the highest average claim amount, and which documents support its top claims?",
        [],
        top_gold,
    )
    add_query(
        "global_top_provider",
        "Find the provider with the highest average claim amount and retrieve evidence for its two largest claims.",
        [],
        top_gold,
    )

    return {
        "metadata": {
            "seed": seed,
            "n_providers": len(providers),
            "n_patients": len(patients),
            "n_claims": len(claims),
            "n_documents": len(documents),
            "n_queries": len(queries),
            "note": "Synthetic data only; not real insurance claims.",
        },
        "providers": providers,
        "patients": patients,
        "claims": claims,
        "documents": documents,
        "payments": payments,
        "queries": queries,
    }


def save_dataset(data: dict[str, Any], path: Path | None = None) -> Path:
    path = path or (DATA_DIR / "synthetic_claims.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    path = path or (DATA_DIR / "synthetic_claims.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    d = generate_dataset()
    p = save_dataset(d)
    print(f"wrote {p}")
    print(d["metadata"])
