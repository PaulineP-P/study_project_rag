from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Корень проекта
BASE_DIR = Path(__file__).parent.parent

# Пути к данным
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = BASE_DIR / "data" / "index"

# Модель эмбеддингов
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# LLM через OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = "google/gemini-3.1-flash-lite-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Параметры retrieval
TOP_K = 3

# System-промпт
SYSTEM_PROMPT = """Ты — ассистент по стратегиям цифровой трансформации регионов России.
Отвечай ТОЛЬКО на основе предоставленного контекста из документов.
Не используй общие знания — только то, что есть в контексте.
В каждом ответе указывай источник: регион и номер страницы.
Если в контексте нет ответа на вопрос, ответь строго:
"В загруженных документах ответ на этот вопрос не найден."
"""
