from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_graphrag.dataset import generate_dataset, save_dataset
from mini_graphrag.graph import build_graph, save_graph_json
from mini_graphrag.text_rag import TextRAG
from mini_graphrag.graph_rag import GraphRAG, HybridRAG
from mini_graphrag.evaluate import evaluate, summarize, write_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini GraphRAG fraud benchmark")
    ap.add_argument("--k", type=int, default=5, help="top-k retrieval")
    ap.add_argument("--seed", type=int, default=42, help="synthetic data seed")
    ap.add_argument("--demo", type=str, default="", help="run one query and show top documents")
    args = ap.parse_args()

    data = generate_dataset(seed=args.seed)
    data_path = save_dataset(data)
    graph = build_graph(data)
    graph_path = save_graph_json(graph)

    text = TextRAG(data["documents"])
    graph_rag = GraphRAG(graph, data["documents"], text)
    methods = {
        "TextRAG": text,
        "GraphRAG": graph_rag,
        "HybridRAG": HybridRAG(graph_rag, text),
    }

    print("Mini GraphRAG Fraud Benchmark")
    print(f"data: {data_path}")
    print(f"graph: {graph_path} | nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(f"documents={len(data['documents'])} queries={len(data['queries'])}")

    if args.demo:
        print("\nDEMO QUERY:", args.demo)
        doc_by_id = {d["doc_id"]: d for d in data["documents"]}
        for method_name, method in methods.items():
            print(f"\n{method_name}:")
            hits = method.search(args.demo, k=args.k)
            for doc_id, score in hits:
                print(f"  {doc_id:35s} score={score:.4f}  {doc_by_id[doc_id]['text'][:140]}")
        return

    rows = evaluate(data["queries"], methods, k=args.k)
    summary = summarize(rows)
    csv_path, md_path, json_path = write_outputs(rows, summary, args.k, data["metadata"])

    print("\nSummary")
    print(f"{'method':10s} {'query_type':28s} {'n':>3s} {'recall':>8s} {'precision':>10s} {'mrr':>8s}")
    for s in summary:
        print(
            f"{s['method']:10s} {s['query_type']:28s} {s['n']:3d} "
            f"{s['recall_at_k']:8.3f} {s['precision_at_k']:10.3f} {s['mrr_at_k']:8.3f}"
        )
    print("\nWrote:")
    print(" ", csv_path)
    print(" ", md_path)
    print(" ", json_path)


if __name__ == "__main__":
    main()
