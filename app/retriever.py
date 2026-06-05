"""
Retrieval: поиск top-k чанков по TF-IDF + cosine similarity.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "data" / "index"


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.vectorizer = None
        self.matrix = None
        self.chunks = []

    def load(self):
        """Загружает индекс из файлов."""
        vectorizer_path = self.index_dir / "vectorizer.pkl"
        matrix_path = self.index_dir / "matrix.npz"
        chunks_path = self.index_dir / "chunks.jsonl"

        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {vectorizer_path}\n"
                "Сначала запусти: uv run python scripts/build_index.py"
            )

        with vectorizer_path.open("rb") as f:
            self.vectorizer = pickle.load(f)

        self.matrix = load_npz(str(matrix_path))

        self.chunks = []
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        return self

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Ищет top_k наиболее релевантных чанков.

        Возвращает список dict с полями:
            text, doc_id, region, page, source_file, score
        """
        if self.vectorizer is None:
            self.load()

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "region": chunk.get("region", ""),
                "page": chunk.get("page", 0),
                "source_file": chunk.get("source_file", ""),
                "score": float(scores[idx]),
            })
        return results
