"""
Построение TF-IDF индекса из documents.jsonl

Шаги:
    1. Ingest: datasets.json → documents.jsonl
    2. Chunking: documents.jsonl → chunks.jsonl
    3. TF-IDF fit: chunks → vectorizer.pkl + matrix.npz

Запуск:
    uv run python scripts/build_index.py
"""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).parent.parent
DATASETS_FILE = BASE_DIR / "data" / "raw" / "datasets.json"
DOCUMENTS_FILE = BASE_DIR / "data" / "processed" / "documents.jsonl"
PROCESSED_CHUNKS = BASE_DIR / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = BASE_DIR / "data" / "index"
INDEX_CHUNKS = INDEX_DIR / "chunks.jsonl"
VECTORIZER_FILE = INDEX_DIR / "vectorizer.pkl"
MATRIX_FILE = INDEX_DIR / "matrix.npz"


# ── Шаг 1: Ingest ─────────────────────────────────────────────────────────────

def ingest(datasets_file: Path, output_file: Path) -> int:
    data = json.loads(datasets_file.read_text(encoding="utf-8"))
    datasets = data["datasets"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for i, item in enumerate(datasets):
            doc = {
                "doc_id": item.get("id", f"doc_{i:04d}"),
                "name": item.get("region", item.get("id", "")),
                "source_file": item.get("source", ""),
                "text": item.get("text", ""),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return len(datasets)


# ── Шаг 2: Chunking ───────────────────────────────────────────────────────────

def chunking(input_file: Path, output_file: Path) -> int:
    from app.chunker import build_chunks
    return build_chunks(input_file, output_file)


# ── Шаг 3: TF-IDF ─────────────────────────────────────────────────────────────

def build_tfidf(chunks_file: Path, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)

    # Читаем чанки
    chunks = []
    with chunks_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]

    # Обучаем TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)

    # Сохраняем артефакты
    with (index_dir / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    save_npz(str(index_dir / "matrix.npz"), matrix)

    # Копируем chunks.jsonl в index/
    shutil.copy(chunks_file, index_dir / "chunks.jsonl")

    return matrix.shape, len(chunks)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Шаг 1
    if not DATASETS_FILE.exists():
        print(f"ERROR: не найден {DATASETS_FILE}")
        print("Сначала запусти: uv run python scripts/prepare_datasets.py")
        return

    print("Шаг 1: Ingest...")
    n_docs = ingest(DATASETS_FILE, DOCUMENTS_FILE)
    print(f"  {n_docs} документов → {DOCUMENTS_FILE}")

    # Шаг 2
    print("\nШаг 2: Chunking...")
    n_chunks = chunking(DOCUMENTS_FILE, PROCESSED_CHUNKS)
    print(f"  {n_chunks} чанков → {PROCESSED_CHUNKS}")

    # Шаг 3
    print("\nШаг 3: TF-IDF индекс...")
    shape, total = build_tfidf(PROCESSED_CHUNKS, INDEX_DIR)
    print(f"  Матрица: {shape[0]} чанков x {shape[1]} признаков")
    print(f"  Сохранено: {VECTORIZER_FILE}")
    print(f"  Сохранено: {MATRIX_FILE}")
    print(f"  Сохранено: {INDEX_CHUNKS}")

    print(f"\n✅ Индекс готов: {total} чанков в {INDEX_DIR}")


if __name__ == "__main__":
    main()
