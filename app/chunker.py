"""
Чанкинг документов из documents.jsonl → chunks.jsonl

Стратегия: разбивка по маркерам страниц "===== СТРАНИЦА N =====" с overlap.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "documents.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"

MAX_CHARS = 2000
OVERLAP = 100
PAGE_MARKER = re.compile(r"===== СТРАНИЦА \d+ =====")


def split_into_pages(text: str) -> list[tuple[int, str]]:
    """Разбивает текст на страницы по маркерам. Возвращает [(page_num, text)]."""
    parts = PAGE_MARKER.split(text)
    markers = PAGE_MARKER.findall(text)

    pages = []
    for i, marker in enumerate(markers):
        page_num = int(re.search(r"\d+", marker).group())
        page_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if page_text:
            pages.append((page_num, page_text))
    return pages


def split_long_page(text: str, max_chars: int, overlap: int) -> list[str]:
    """Разбивает длинную страницу на части с перекрытием."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """Разбивает один документ на чанки."""
    text = doc.get("text", "")
    doc_id = doc.get("doc_id", "unknown")
    source_file = doc.get("source_file", "")
    name = doc.get("name", "")

    pages = split_into_pages(text)

    # Если маркеров страниц нет — обрабатываем весь текст как одну страницу
    if not pages:
        pages = [(1, text.strip())]

    chunks = []
    chunk_index = 0
    for page_num, page_text in pages:
        parts = split_long_page(page_text, MAX_CHARS, OVERLAP)
        for part in parts:
            if not part.strip():
                continue
            chunks.append({
                "chunk_id": f"{doc_id}_p{page_num}_{chunk_index}",
                "doc_id": doc_id,
                "source_file": source_file,
                "region": name,
                "page": page_num,
                "text": part.strip(),
            })
            chunk_index += 1
    return chunks


def build_chunks(input_file: Path, output_file: Path) -> int:
    """Читает documents.jsonl, чанкует, пишет chunks.jsonl."""
    if not input_file.exists():
        raise FileNotFoundError(f"Не найден {input_file}. Сначала запусти ingest.py")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with output_file.open("w", encoding="utf-8") as out:
        with input_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                chunks = chunk_document(doc)
                for chunk in chunks:
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total += 1
                print(f"  {doc.get('region', doc.get('doc_id'))}: {len(chunks)} чанков")

    return total


if __name__ == "__main__":
    print(f"Чанкинг: {INPUT_FILE} → {OUTPUT_FILE}")
    total = build_chunks(INPUT_FILE, OUTPUT_FILE)
    print(f"\nГотово: {total} чанков → {OUTPUT_FILE}")
