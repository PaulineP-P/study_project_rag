"""
Промпты для RAG-генератора.
"""

SYSTEM_PROMPT = """Ты — ассистент по стратегиям цифровой трансформации регионов России.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста из документов.
2. Не используй общие знания — только то, что есть в контексте.
3. В каждом ответе указывай источник: регион и номер страницы в формате [Регион, стр. N].
4. Если в контексте нет ответа, ответь строго и только:
   "В загруженных документах ответ на этот вопрос не найден."
5. Не додумывай и не интерпретируй сверх написанного в документах.
"""

NO_CONTEXT_THRESHOLD = 0.05  # score ниже этого — контекст считается нерелевантным


def build_context(chunks: list[dict]) -> str:
    """Формирует текстовый контекст из найденных чанков."""
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, 1):
        region = chunk.get("region", chunk.get("doc_id", "Неизвестно"))
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[Источник {i}: {region}, стр. {page}]\n{text}")

    return "\n\n".join(parts)


def build_user_message(query: str, chunks: list[dict]) -> str:
    """Формирует сообщение пользователя с контекстом и вопросом."""
    context = build_context(chunks)

    if not context:
        return f"Вопрос: {query}\n\nКонтекст: отсутствует."

    return f"Контекст из документов:\n\n{context}\n\n---\nВопрос: {query}"


def is_relevant(chunks: list[dict], threshold: float = NO_CONTEXT_THRESHOLD) -> bool:
    """Проверяет есть ли хоть один релевантный чанк."""
    if not chunks:
        return False
    return any(chunk.get("score", 0) >= threshold for chunk in chunks)
