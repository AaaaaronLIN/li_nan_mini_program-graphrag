"""A tiny directed graph implementation (no external dependencies)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.inc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.edge_count = 0

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes[node_id] = attrs

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        self.out[src].append((dst, relation))
        self.inc[dst].append((src, relation))
        self.edge_count += 1

    def node_type(self, node_id: str) -> str | None:
        return self.nodes.get(node_id, {}).get("type")

    def successors(self, node_id: str, relation: str | None = None) -> list[str]:
        return [dst for dst, rel in self.out.get(node_id, []) if relation is None or rel == relation]

    def predecessors(self, node_id: str, relation: str | None = None) -> list[str]:
        return [src for src, rel in self.inc.get(node_id, []) if relation is None or rel == relation]

    def neighbors(self, node_id: str) -> list[str]:
        return list(dict.fromkeys(self.successors(node_id) + self.predecessors(node_id)))

    def nodes_of_type(self, node_type: str) -> list[str]:
        return [n for n, attrs in self.nodes.items() if attrs.get("type") == node_type]

    def subgraph_edges(self, seeds: list[str], hops: int = 2) -> list[tuple[str, str, str]]:
        seen = set(seeds)
        frontier = list(seeds)
        for _ in range(hops):
            nxt = []
            for node in frontier:
                for neighbor in self.neighbors(node):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        nxt.append(neighbor)
            frontier = nxt
        edges = []
        for src in seen:
            for dst, rel in self.out.get(src, []):
                if dst in seen:
                    edges.append((src, dst, rel))
        return edges

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_edges(self) -> int:
        return self.edge_count


def build_graph(data: dict[str, Any]) -> Graph:
    g = Graph()
    for p in data["providers"]:
        g.add_node(p["provider_id"], type="Provider", **p)
    for s in data["patients"]:
        g.add_node(s["patient_id"], type="Patient", **s)
    for c in data["claims"]:
        g.add_node(c["claim_id"], type="Claim", **c)
        g.add_edge(c["patient_id"], c["provider_id"], "treated_by")
        g.add_edge(c["provider_id"], c["claim_id"], "billed")
        g.add_edge(c["claim_id"], c["patient_id"], "has_patient")
    for d in data["documents"]:
        g.add_node(d["doc_id"], type="Document", **d)
        g.add_edge(d["claim_id"], d["doc_id"], "supported_by")
    for pay in data["payments"]:
        g.add_node(pay["payment_id"], type="Payment", **pay)
        g.add_edge(pay["payment_id"], pay["claim_id"], "pays")
    return g


def save_graph_json(g: Graph, path: Path | None = None) -> Path:
    path = path or (DATA_DIR / "graph.json")
    payload = {
        "nodes": g.nodes,
        "edges": [{"src": s, "dst": d, "relation": r} for s, items in g.out.items() for d, r in items],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
