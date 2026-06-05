# RAG-ассистент: Цифровая трансформация регионов РФ

RAG-система для поиска ответов по стратегиям цифровой трансформации субъектов РФ.
Отвечает на вопросы только по загруженным документам, указывая источник (регион и страницу).

## Скриншоты

> _Добавить скриншоты интерфейса после запуска_

## Стек

| Компонент | Решение |
|-----------|---------|
| Язык | Python 3.11+ |
| Индекс | TF-IDF (scikit-learn) + cosine similarity |
| UI | Streamlit |
| LLM | Google Gemini 3.1 Flash Lite через OpenRouter |
| Зависимости | uv |

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/PaulineP-P/study_project_rag.git
cd study_project_rag
```

### 2. Установить зависимости

```bash
uv venv && uv sync
```

### 3. Настроить API ключ

```bash
cp .env.example .env
```

Открой `.env` и вставь ключ от [OpenRouter](https://openrouter.ai/keys):

```
OPENROUTER_API_KEY=your_key_here
```

### 4. Подготовить данные

Положи `.txt` файлы стратегий цифровой трансформации в папку `data/raw/`:

```bash
cp /path/to/your/files/*.txt data/raw/
```

Затем запусти подготовку данных:

```bash
uv run python scripts/prepare_datasets.py
```

### 5. Построить индекс

```bash
uv run python scripts/build_index.py
```

Ожидаемый вывод:
```
Шаг 1: Ingest...   87 документов
Шаг 2: Chunking... 10000+ чанков
Шаг 3: TF-IDF индекс... Матрица: XXXX чанков x XXXX признаков
✅ Индекс готов
```

### 6. Запустить интерфейс

```bash
uv run streamlit run app/main.py
```

Открой в браузере: http://localhost:8501

## Структура проекта

```
├── app/
│   ├── chunker.py       # разбивка документов на чанки
│   ├── config.py        # конфигурация
│   ├── generator.py     # генерация ответа через LLM
│   ├── main.py          # Streamlit UI
│   ├── prompts.py       # системный промпт и контекст
│   └── retriever.py     # поиск top-k чанков
├── scripts/
│   ├── build_index.py       # построение TF-IDF индекса
│   ├── check_generator.py   # проверка генератора
│   ├── check_retrieval.py   # проверка поиска
│   ├── ingest.py            # конвертация datasets.json → documents.jsonl
│   └── prepare_datasets.py  # подготовка данных из .txt файлов
├── tests/
│   ├── test_chunking.py     # 13 тестов чанкера
│   └── test_retrieval.py    # 10 тестов retriever
├── doc/
│   ├── 00_project_idea.md
│   ├── conventions.md
│   ├── tasklist.md
│   ├── vision.md
│   └── workflow.md
├── data/
│   ├── raw/             # исходные .txt файлы (не в git)
│   ├── processed/       # промежуточные файлы (не в git)
│   └── index/           # TF-IDF индекс (не в git)
├── .env.example
└── pyproject.toml
```

## Тесты

```bash
uv run pytest tests/ -v
```

Ожидаемый результат: **23 passed**

## Demo-вопросы

| Вопрос | Ожидаемый результат |
|--------|---------------------|
| Какие задачи цифровой трансформации в здравоохранении? | Ответ с источником |
| Какие проблемы в образовании решает цифровизация? | Ответ с источником |
| Какие технологии внедряются в госуправлении? | Ответ с источником |
| Как готовить борщ? | Явный отказ |

## Улучшения (план)

- **Семантический чанкинг** — разбивка по смыслу вместо страниц
- **Reranking** — дополнительная сортировка результатов после retrieval
- **Многоязычные эмбеддинги** — замена TF-IDF на `multilingual-e5-small`
- **Инкрементальное обновление индекса** — добавление новых документов без полной перестройки
- **Кэширование запросов** — ускорение повторных обращений
