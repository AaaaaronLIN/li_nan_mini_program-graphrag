"""Retrieval metrics and report writers."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(gold)) / max(1, len(gold))


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(gold)) / max(1, k)


def mrr_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


def evaluate(queries: list[dict[str, Any]], methods: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    rows = []
    for q in queries:
        for method_name, method in methods.items():
            if getattr(method, "accepts_seeds", False):
                hits = method.search(q["question"], k=k, seeds=q.get("seed_entities", []))
            else:
                hits = method.search(q["question"], k=k)
            retrieved = [doc_id for doc_id, _ in hits]
            rows.append({
                "query_id": q["query_id"],
                "query_type": q["query_type"],
                "question": q["question"],
                "method": method_name,
                "recall_at_k": round(recall_at_k(retrieved, q["gold_doc_ids"], k), 4),
                "precision_at_k": round(precision_at_k(retrieved, q["gold_doc_ids"], k), 4),
                "mrr_at_k": round(mrr_at_k(retrieved, q["gold_doc_ids"], k), 4),
                "retrieved_doc_ids": ";".join(retrieved),
                "gold_doc_ids": ";".join(q["gold_doc_ids"]),
            })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"n": 0, "recall": 0.0, "precision": 0.0, "mrr": 0.0})
    for r in rows:
        key = (r["method"], r["query_type"])
        g = grouped[key]
        g["n"] += 1
        g["recall"] += r["recall_at_k"]
        g["precision"] += r["precision_at_k"]
        g["mrr"] += r["mrr_at_k"]
    out = []
    for (method, query_type), g in sorted(grouped.items()):
        out.append({
            "method": method,
            "query_type": query_type,
            "n": int(g["n"]),
            "recall_at_k": round(g["recall"] / g["n"], 4),
            "precision_at_k": round(g["precision"] / g["n"], 4),
            "mrr_at_k": round(g["mrr"] / g["n"], 4),
        })
    return out


def write_outputs(rows: list[dict[str, Any]], summary: list[dict[str, Any]], k: int, metadata: dict[str, Any]) -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "results.csv"
    md_path = OUT_DIR / "results.md"
    json_path = OUT_DIR / "summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Mini GraphRAG Fraud  Results (k={k})",
        "",
        f"- Documents: {metadata['n_documents']}",
        f"- Queries: {metadata['n_queries']}",
        f"- Claims: {metadata['n_claims']}",
        "",
        "| Method | Query type | n | Recall@k | Precision@k | MRR@k |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['method']} | {s['query_type']} | {s['n']} | "
            f"{s['recall_at_k']:.4f} | {s['precision_at_k']:.4f} | {s['mrr_at_k']:.4f} |"
        )
    lines += [
        "",
        "> Retrieval-only benchmark on synthetic data. No LLM generation or human evaluation yet.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "k": k, "summary": summary}, f, ensure_ascii=False, indent=2)

    return csv_path, md_path, json_path
