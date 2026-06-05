"""
Генератор ответов через OpenRouter API (совместим с OpenAI SDK).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.prompts import SYSTEM_PROMPT, build_user_message, is_relevant

load_dotenv()

LLM_MODEL = "google/gemini-3.1-flash-lite-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NO_ANSWER = "В загруженных документах ответ на этот вопрос не найден."


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY не задан. "
            "Создай файл .env с ключом: OPENROUTER_API_KEY=your_key_here"
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Генерирует ответ на основе найденных чанков.

    Возвращает dict:
        answer  — текст ответа
        sources — список источников (region, page, score)
        refused — True если система отказала (нет релевантного контекста)
    """
    # Если нет релевантного контекста — отказываем сразу без вызова LLM
    if not is_relevant(chunks):
        return {
            "answer": NO_ANSWER,
            "sources": [],
            "refused": True,
        }

    client = get_client()
    user_message = build_user_message(query, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content.strip()

    sources = [
        {
            "region": c.get("region", c.get("doc_id", "")),
            "page": c.get("page", "?"),
            "score": round(c.get("score", 0), 4),
        }
        for c in chunks
        if c.get("score", 0) > 0
    ]

    return {
        "answer": answer,
        "sources": sources,
        "refused": False,
    }
