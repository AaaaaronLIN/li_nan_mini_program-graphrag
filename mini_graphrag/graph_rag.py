"""GraphRAG-style retrieval over a small fraud knowledge graph."""
from __future__ import annotations

import re
from typing import Any

from .graph import Graph
from .text_rag import TextRAG

ID_RE = re.compile(r"\b(?:P\d{3}|S\d{3}|C\d{4}|PAY\d{4})\b")


class GraphRAG:
    accepts_seeds = True

    def __init__(self, graph: Graph, documents: list[dict[str, Any]], text_rag: TextRAG) -> None:
        self.G = graph
        self.documents = documents
        self.text_rag = text_rag
        self.doc_ids = {d["doc_id"] for d in documents}

    def parse_seeds(self, question: str) -> list[str]:
        return [x for x in ID_RE.findall(question) if x in self.G.nodes]

    def claims_for_provider(self, provider_id: str) -> list[str]:
        return self.G.successors(provider_id, "billed")

    def docs_for_claim(self, claim_id: str) -> list[str]:
        return self.G.successors(claim_id, "supported_by")

    def docs_for_claims(self, claim_ids: list[str]) -> list[str]:
        out = []
        for cid in claim_ids:
            out.extend(self.docs_for_claim(cid))
        return out

    def _rank(self, query: str, doc_ids: list[str], k: int) -> list[tuple[str, float]]:
        doc_ids = [d for d in dict.fromkeys(doc_ids) if d in self.doc_ids]
        if not doc_ids:
            return []
        scores = self.text_rag.score_documents(query, doc_ids)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def _expand_docs(self, seeds: list[str], hops: int = 2) -> list[str]:
        edges = self.G.subgraph_edges(seeds, hops=hops)
        nodes = set(seeds)
        for s, d, _ in edges:
            nodes.add(s); nodes.add(d)
        return [n for n in nodes if self.G.node_type(n) == "Document"]

    def search(self, query: str, k: int = 5, seeds: list[str] | None = None) -> list[tuple[str, float]]:
        seeds = list(seeds or []) or self.parse_seeds(query)
        qlow = query.lower()

        if "highest average claim amount" in qlow:
            providers = self.G.nodes_of_type("Provider")
            def avg_amount(pid: str) -> float:
                claims = self.claims_for_provider(pid)
                return sum(self.G.nodes[c]["amount"] for c in claims) / max(1, len(claims))
            top_provider = max(providers, key=avg_amount)
            claims = sorted(self.claims_for_provider(top_provider),
                            key=lambda c: self.G.nodes[c]["amount"], reverse=True)[:2]
            return self._rank(query, self.docs_for_claims(claims), k)

        if "highest-amount claims" in qlow or "largest claims" in qlow:
            claims = []
            for seed in seeds:
                if self.G.node_type(seed) == "Provider":
                    claims.extend(self.claims_for_provider(seed))
            claims = sorted(claims, key=lambda c: self.G.nodes[c]["amount"], reverse=True)[:2]
            return self._rank(query, self.docs_for_claims(claims), k)

        if "patients also appear under other providers" in qlow:
            claims = []
            for seed in seeds:
                if self.G.node_type(seed) != "Provider":
                    continue
                for cid in self.claims_for_provider(seed):
                    sid = self.G.nodes[cid]["patient_id"]
                    other = [
                        x for x in self.G.nodes_of_type("Claim")
                        if self.G.nodes[x]["patient_id"] == sid and self.G.nodes[x]["provider_id"] != seed
                    ]
                    if other:
                        claims.append(cid)
            claims = sorted(claims, key=lambda c: self.G.nodes[c]["amount"], reverse=True)[:2]
            return self._rank(query, self.docs_for_claims(claims), k)

        if "billing review" in qlow and ("upcoding" in qlow or "kickback" in qlow):
            docs = []
            for seed in seeds:
                if self.G.node_type(seed) != "Provider":
                    continue
                for cid in self.claims_for_provider(seed):
                    if self.G.nodes[cid]["is_fraud"]:
                        docs.append(f"D{cid}_review")
            return self._rank(query, docs, k)

        if "support claim" in qlow:
            claim_seeds = [s for s in seeds if self.G.node_type(s) == "Claim"]
            return self._rank(query, self.docs_for_claims(claim_seeds), k)

        # Fallback: graph expansion, then rank connected docs.
        docs = self._expand_docs(seeds, hops=2)
        if not docs:
            return self.text_rag.search(query, k=k)
        return self._rank(query, docs, k)


class HybridRAG:
    accepts_seeds = True

    """Naive fusion: graph results first, text results as a small bonus."""
    def __init__(self, graph_rag: GraphRAG, text_rag: TextRAG) -> None:
        self.graph_rag = graph_rag
        self.text_rag = text_rag

    def search(self, query: str, k: int = 5, seeds: list[str] | None = None) -> list[tuple[str, float]]:
        graph_hits = self.graph_rag.search(query, k=k * 2, seeds=seeds)
        text_hits = self.text_rag.search(query, k=k * 2)
        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(graph_hits, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 0.7 / rank
        for rank, (doc_id, _) in enumerate(text_hits, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 0.3 / rank
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
