"""
Подготовка data/raw/datasets.json из .txt файлов в data/raw/

Запуск:
    uv run python scripts/prepare_datasets.py

Ожидает что .txt файлы уже лежат в data/raw/
"""

import json
import sys
from pathlib import Path

# Корень проекта — папка выше scripts/
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "datasets.json"


def extract_region_name(filename: str) -> str:
    """Извлекает читаемое название региона из имени файла."""
    name = filename.replace("_ocr.txt", "").replace(".txt", "")
    name = name.replace("-", " ").replace("_", " ")
    return name.title()


def load_txt_files(raw_dir: Path) -> list[dict]:
    """Читает все .txt файлы и возвращает список записей."""
    txt_files = sorted(raw_dir.glob("*.txt"))

    if not txt_files:
        print(f"ERROR: нет .txt файлов в {raw_dir}")
        sys.exit(1)

    datasets = []
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            print(f"  пропускаю пустой файл: {path.name}")
            continue

        datasets.append({
            "id": path.stem,
            "source": path.name,
            "region": extract_region_name(path.name),
            "text": text,
        })
        print(f"  загружен: {path.name} ({len(text):,} символов)")

    return datasets


def main():
    print(f"Читаю .txt файлы из: {RAW_DIR}")
    datasets = load_txt_files(RAW_DIR)

    output = {"datasets": datasets}
    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nГотово: {len(datasets)} документов → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
