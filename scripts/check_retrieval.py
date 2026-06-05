"""
Проверка retrieval: релевантный и нерелевантный запросы.

Запуск:
    uv run python scripts/check_retrieval.py
"""

from app.retriever import Retriever

RELEVANT_QUERY = "цифровая трансформация государственного управления"
IRRELEVANT_QUERY = "рецепт борща с грибами"


def check(retriever: Retriever, query: str, label: str):
    print(f"\n{'─' * 60}")
    print(f"[{label}] Запрос: «{query}»")
    results = retriever.search(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  #{i} score={r['score']:.4f} | {r['region']} стр.{r['page']}")
        print(f"      {r['text'][:120].strip()}...")
    return results


def main():
    print("Загрузка индекса...")
    retriever = Retriever()
    retriever.load()
    print(f"Загружено {len(retriever.chunks)} чанков")

    rel = check(retriever, RELEVANT_QUERY, "РЕЛЕВАНТНЫЙ")
    irrel = check(retriever, IRRELEVANT_QUERY, "НЕРЕЛЕВАНТНЫЙ")

    print(f"\n{'─' * 60}")
    top_rel_score = rel[0]["score"] if rel else 0
    top_irrel_score = irrel[0]["score"] if irrel else 0

    print(f"Релевантный запрос — top score:    {top_rel_score:.4f}")
    print(f"Нерелевантный запрос — top score:  {top_irrel_score:.4f}")

    assert top_rel_score > 0, "❌ Релевантный запрос должен вернуть score > 0"
    assert top_rel_score > top_irrel_score, "❌ Релевантный запрос должен быть выше нерелевантного"
    print("\n✅ Retrieval работает корректно")


if __name__ == "__main__":
    main()
