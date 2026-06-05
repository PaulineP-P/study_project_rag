"""Тесты для app/chunker.py"""

import json
import tempfile
from pathlib import Path

import pytest

from app.chunker import chunk_document, split_into_pages, split_long_page, build_chunks


# ── Фикстуры ──────────────────────────────────────────────────────────────────

SAMPLE_DOC = {
    "doc_id": "test_region",
    "name": "Тестовый регион",
    "source_file": "test_region_ocr.txt",
    "text": (
        "===== СТРАНИЦА 1 =====\n"
        "Первый абзац страницы один. Содержит информацию о цифровизации.\n\n"
        "===== СТРАНИЦА 2 =====\n"
        "Второй абзац страницы два. Информация о государственном управлении.\n\n"
        "===== СТРАНИЦА 3 =====\n"
        "Третий абзац страницы три. Данные о здравоохранении и образовании.\n"
    ),
}

EMPTY_DOC = {
    "doc_id": "empty_doc",
    "name": "Пустой документ",
    "source_file": "empty.txt",
    "text": "",
}


# ── Тесты split_into_pages ────────────────────────────────────────────────────

def test_split_into_pages_count():
    """Корректное количество страниц."""
    pages = split_into_pages(SAMPLE_DOC["text"])
    assert len(pages) == 3


def test_split_into_pages_numbers():
    """Номера страниц корректны."""
    pages = split_into_pages(SAMPLE_DOC["text"])
    numbers = [p[0] for p in pages]
    assert numbers == [1, 2, 3]


def test_split_into_pages_no_markers():
    """Текст без маркеров возвращает пустой список."""
    pages = split_into_pages("Просто текст без маркеров страниц.")
    assert pages == []


# ── Тесты split_long_page ─────────────────────────────────────────────────────

def test_split_long_page_short_text():
    """Короткий текст не разбивается."""
    text = "Короткий текст"
    result = split_long_page(text, max_chars=100, overlap=10)
    assert result == [text]


def test_split_long_page_splits_correctly():
    """Длинный текст разбивается на несколько частей."""
    text = "А" * 500
    result = split_long_page(text, max_chars=200, overlap=20)
    assert len(result) > 1


def test_split_long_page_no_empty_chunks():
    """Не создаёт пустых частей."""
    text = "Б" * 300
    result = split_long_page(text, max_chars=100, overlap=10)
    assert all(len(c) > 0 for c in result)


# ── Тесты chunk_document ──────────────────────────────────────────────────────

def test_chunk_document_returns_list():
    """Результат — список."""
    chunks = chunk_document(SAMPLE_DOC)
    assert isinstance(chunks, list)


def test_chunk_document_not_empty():
    """Непустой документ даёт чанки."""
    chunks = chunk_document(SAMPLE_DOC)
    assert len(chunks) > 0


def test_chunk_document_has_required_fields():
    """Каждый чанк содержит обязательные поля."""
    chunks = chunk_document(SAMPLE_DOC)
    required = {"chunk_id", "doc_id", "source_file", "region", "page", "text"}
    for chunk in chunks:
        assert required.issubset(chunk.keys()), f"Отсутствуют поля: {required - chunk.keys()}"


def test_chunk_document_no_empty_text():
    """Нет чанков с пустым текстом."""
    chunks = chunk_document(SAMPLE_DOC)
    for chunk in chunks:
        assert chunk["text"].strip() != "", "Найден чанк с пустым текстом"


def test_chunk_document_doc_id_propagated():
    """doc_id передаётся в каждый чанк."""
    chunks = chunk_document(SAMPLE_DOC)
    for chunk in chunks:
        assert chunk["doc_id"] == "test_region"


def test_chunk_document_empty_text():
    """Пустой документ не создаёт чанков с пустым текстом."""
    chunks = chunk_document(EMPTY_DOC)
    for chunk in chunks:
        assert chunk["text"].strip() != ""


# ── Тест build_chunks (интеграционный) ───────────────────────────────────────

def test_build_chunks_creates_output():
    """build_chunks создаёт файл с чанками."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_file = tmpdir / "documents.jsonl"
        output_file = tmpdir / "chunks.jsonl"

        # Записываем тестовый документ
        with input_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(SAMPLE_DOC, ensure_ascii=False) + "\n")

        total = build_chunks(input_file, output_file)

        assert output_file.exists()
        assert total > 0

        # Проверяем что файл читается
        with output_file.open(encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == total
