"""
Конвертация data/raw/datasets.json → data/processed/documents.jsonl

Запуск:
    uv run python scripts/ingest.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "datasets.json"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "documents.jsonl"


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: не найден {INPUT_FILE}")
        print("Сначала запусти: uv run python scripts/prepare_datasets.py")
        return

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    datasets = data["datasets"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for i, item in enumerate(datasets):
            doc = {
                "doc_id": item.get("id", f"doc_{i:04d}"),
                "name": item.get("region", item.get("id", f"doc_{i:04d}")),
                "source_file": item.get("source", ""),
                "text": item.get("text", ""),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print(f"Готово: {count} документов → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
