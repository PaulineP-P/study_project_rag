"""
Проверка генератора: релевантный и нерелевантный запросы.

Запуск:
    uv run python scripts/check_generator.py
"""

from app.retriever import Retriever
from app.generator import generate_answer

RELEVANT_QUERY = "Какие задачи цифровой трансформации в сфере здравоохранения?"
IRRELEVANT_QUERY = "Как приготовить борщ?"


def check(retriever: Retriever, query: str, label: str):
    print(f"\n{'─' * 60}")
    print(f"[{label}] Запрос: «{query}»")

    chunks = retriever.search(query, top_k=3)
    result = generate_answer(query, chunks)

    print(f"\nОтвет:\n{result['answer']}")

    if result["sources"]:
        print("\nИсточники:")
        for s in result["sources"]:
            print(f"  • {s['region']}, стр. {s['page']} (score={s['score']})")

    print(f"\nОтказ: {result['refused']}")
    return result


def main():
    print("Загрузка индекса...")
    retriever = Retriever()
    retriever.load()
    print(f"Загружено {len(retriever.chunks)} чанков")

    rel = check(retriever, RELEVANT_QUERY, "РЕЛЕВАНТНЫЙ")
    irrel = check(retriever, IRRELEVANT_QUERY, "НЕРЕЛЕВАНТНЫЙ")

    print(f"\n{'─' * 60}")
    assert not rel["refused"], "❌ Релевантный запрос не должен отказывать"
    assert irrel["refused"], "❌ Нерелевантный запрос должен давать отказ"
    assert rel["sources"], "❌ Релевантный ответ должен содержать источники"

    print("✅ Генератор работает корректно")


if __name__ == "__main__":
    main()
