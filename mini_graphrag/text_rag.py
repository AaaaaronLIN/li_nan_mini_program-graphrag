"""TF-IDF text retrieval in pure Python."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class TextRAG:
    accepts_seeds = False

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.doc_ids = [d["doc_id"] for d in documents]
        self.doc_by_id = {d["doc_id"]: d for d in documents}
        self.doc_index = {d["doc_id"]: i for i, d in enumerate(documents)}
        self.doc_tokens = [tokenize(d["text"]) for d in documents]
        self.doc_tfs = [Counter(tokens) for tokens in self.doc_tokens]
        self.df: Counter[str] = Counter()
        for tf in self.doc_tfs:
            self.df.update(tf.keys())
        self.n_docs = len(documents)
        self.idf = {
            term: math.log((1 + self.n_docs) / (1 + df)) + 1.0
            for term, df in self.df.items()
        }
        self.doc_vectors = [self._vectorize_tf(tf) for tf in self.doc_tfs]
        self.doc_norms = [self._norm(v) for v in self.doc_vectors]

    def _vectorize_tf(self, tf: Counter[str]) -> dict[str, float]:
        vec = {}
        for term, count in tf.items():
            if term in self.idf:
                vec[term] = (1 + math.log(count)) * self.idf[term]
        return vec

    @staticmethod
    def _norm(vec: dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values())) or 1.0

    def _query_vector(self, query: str) -> dict[str, float]:
        return self._vectorize_tf(Counter(tokenize(query)))

    def score_documents(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        qvec = self._query_vector(query)
        qnorm = self._norm(qvec)
        out = {}
        for doc_id in doc_ids:
            i = self.doc_index[doc_id]
            dvec, dnorm = self.doc_vectors[i], self.doc_norms[i]
            small, large = (qvec, dvec) if len(qvec) < len(dvec) else (dvec, qvec)
            dot = sum(weight * large.get(term, 0.0) for term, weight in small.items())
            out[doc_id] = dot / (qnorm * dnorm)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        qvec = self._query_vector(query)
        qnorm = self._norm(qvec)
        scored = []
        for doc_id, dvec, dnorm in zip(self.doc_ids, self.doc_vectors, self.doc_norms):
            # sparse dot product
            small, large = (qvec, dvec) if len(qvec) < len(dvec) else (dvec, qvec)
            dot = sum(weight * large.get(term, 0.0) for term, weight in small.items())
            scored.append((doc_id, dot / (qnorm * dnorm)))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
