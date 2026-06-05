"""Тесты для app/retriever.py"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retriever import Retriever


# ── Фикстура: мини-индекс ─────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "chunk_id": "doc1_p1_0",
        "doc_id": "altai",
        "region": "Алтайский край",
        "page": 1,
        "source_file": "altai_ocr.txt",
        "text": "Цифровая трансформация здравоохранения включает внедрение телемедицины и электронных карт пациентов.",
    },
    {
        "chunk_id": "doc1_p2_0",
        "doc_id": "altai",
        "region": "Алтайский край",
        "page": 2,
        "source_file": "altai_ocr.txt",
        "text": "Государственное управление: электронный документооборот и цифровые платформы для госуслуг.",
    },
    {
        "chunk_id": "doc1_p3_0",
        "doc_id": "altai",
        "region": "Алтайский край",
        "page": 3,
        "source_file": "altai_ocr.txt",
        "text": "Образование: цифровая образовательная среда и онлайн-обучение в школах и вузах.",
    },
    {
        "chunk_id": "doc1_p4_0",
        "doc_id": "altai",
        "region": "Алтайский край",
        "page": 4,
        "source_file": "altai_ocr.txt",
        "text": "Сельское хозяйство: цифровые платформы АПК и мониторинг сельскохозяйственных угодий.",
    },
    {
        "chunk_id": "doc1_p5_0",
        "doc_id": "altai",
        "region": "Алтайский край",
        "page": 5,
        "source_file": "altai_ocr.txt",
        "text": "Транспорт и логистика: интеллектуальные транспортные системы и ГЛОНАСС.",
    },
]


@pytest.fixture
def temp_index(tmp_path):
    """Создаёт временный TF-IDF индекс из sample chunks."""
    texts = [c["text"] for c in SAMPLE_CHUNKS]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    with (tmp_path / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    save_npz(str(tmp_path / "matrix.npz"), matrix)

    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in SAMPLE_CHUNKS:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return tmp_path


@pytest.fixture
def retriever(temp_index):
    """Возвращает загруженный Retriever с тестовым индексом."""
    r = Retriever(index_dir=temp_index)
    r.load()
    return r


# ── Тесты загрузки ────────────────────────────────────────────────────────────

def test_retriever_loads(retriever):
    """Retriever загружается без ошибок."""
    assert retriever.vectorizer is not None
    assert retriever.matrix is not None
    assert len(retriever.chunks) == len(SAMPLE_CHUNKS)


def test_retriever_chunks_count(retriever):
    """Количество чанков совпадает с исходными данными."""
    assert len(retriever.chunks) == 5


def test_retriever_missing_index_raises(tmp_path):
    """Ошибка если индекс не найден."""
    r = Retriever(index_dir=tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        r.load()


# ── Тесты поиска ──────────────────────────────────────────────────────────────

def test_search_returns_list(retriever):
    """search возвращает список."""
    results = retriever.search("здравоохранение", top_k=3)
    assert isinstance(results, list)


def test_search_returns_top_k(retriever):
    """search возвращает не больше top_k результатов."""
    results = retriever.search("цифровая трансформация", top_k=3)
    assert len(results) <= 3


def test_search_relevant_has_positive_score(retriever):
    """Релевантный запрос даёт score > 0."""
    results = retriever.search("телемедицины электронных карт пациентов", top_k=3)
    assert results[0]["score"] > 0, "Релевантный запрос должен давать score > 0"


def test_search_irrelevant_has_zero_score(retriever):
    """Нерелевантный запрос даёт score = 0."""
    results = retriever.search("рецепт борща с грибами", top_k=3)
    assert results[0]["score"] == 0.0, "Нерелевантный запрос должен давать score = 0"


def test_search_result_fields(retriever):
    """Каждый результат содержит обязательные поля."""
    results = retriever.search("государственное управление", top_k=2)
    required = {"text", "doc_id", "region", "page", "source_file", "score"}
    for r in results:
        assert required.issubset(r.keys()), f"Отсутствуют поля: {required - r.keys()}"


def test_search_relevance_order(retriever):
    """Результаты отсортированы по убыванию score."""
    results = retriever.search("образование школы обучение", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Результаты должны быть по убыванию score"


def test_search_top1_most_relevant(retriever):
    """Первый результат наиболее релевантен запросу."""
    results = retriever.search("телемедицина электронные карты пациентов", top_k=3)
    assert results[0]["score"] >= results[1]["score"]
