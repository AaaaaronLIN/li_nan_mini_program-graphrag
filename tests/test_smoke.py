import unittest
from mini_graphrag.dataset import generate_dataset
from mini_graphrag.graph import build_graph
from mini_graphrag.text_rag import TextRAG
from mini_graphrag.graph_rag import GraphRAG


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset(seed=42)
        cls.graph = build_graph(cls.data)
        cls.text = TextRAG(cls.data["documents"])
        cls.graph_rag = GraphRAG(cls.graph, cls.data["documents"], cls.text)

    def test_counts(self):
        self.assertEqual(len(self.data["providers"]), 30)
        self.assertEqual(len(self.data["claims"]), 300)
        self.assertEqual(len(self.data["documents"]), 900)
        self.assertEqual(len(self.data["queries"]), 30)

    def test_graph(self):
        self.assertEqual(self.graph.number_of_nodes(), 1650)
        self.assertEqual(self.graph.number_of_edges(), 2100)

    def test_graph_rag_single_claim(self):
        hits = self.graph_rag.search("Which documents support claim C0001?", k=5)
        ids = [x[0] for x in hits]
        self.assertIn("DC0001_summary", ids)
        self.assertIn("DC0001_note", ids)
        self.assertIn("DC0001_review", ids)

    def test_graph_rag_provider_high_amount(self):
        hits = self.graph_rag.search(
            "Which documents support the highest-amount claims billed by provider P001?",
            k=5,
        )
        self.assertGreaterEqual(len(hits), 5)


if __name__ == "__main__":
    unittest.main()
