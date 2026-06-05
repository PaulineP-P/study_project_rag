"""
Streamlit UI для RAG-ассистента по стратегиям цифровой трансформации регионов РФ.

Запуск:
    uv run streamlit run app/main.py
"""

import os
import streamlit as st
from pathlib import Path

# ── Конфигурация страницы ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG: Цифровая трансформация регионов РФ",
    page_icon="🏛️",
    layout="wide",
)

# ── Demo-вопросы ───────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Какие задачи цифровой трансформации в сфере здравоохранения?",
    "Какие проблемы в сфере образования решает цифровизация?",
    "Какие технологии внедряются в государственном управлении?",
    "Как готовить борщ?",  # negative — должен получить отказ
]

# ── Загрузка индекса ───────────────────────────────────────────────────────────

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


@st.cache_resource(show_spinner="Загрузка индекса...")
def load_retriever():
    from app.retriever import Retriever
    r = Retriever()
    r.load()
    return r


def index_exists() -> bool:
    return (INDEX_DIR / "vectorizer.pkl").exists()


# ── Основной UI ────────────────────────────────────────────────────────────────

st.title("🏛️ RAG-ассистент: Цифровая трансформация регионов РФ")
st.caption("Ответы только по загруженным стратегиям цифровой трансформации субъектов РФ")

# Проверка индекса
if not index_exists():
    st.error(
        "⚠️ Индекс не найден. Сначала запусти в терминале:\n\n"
        "```bash\n"
        "uv run python scripts/prepare_datasets.py\n"
        "uv run python scripts/build_index.py\n"
        "```"
    )
    st.stop()

# Загружаем retriever
retriever = load_retriever()
st.success(f"✅ Индекс загружен: {len(retriever.chunks)} чанков из документов")

st.divider()

# ── Demo-вопросы (кнопки) ──────────────────────────────────────────────────────

st.markdown("**Demo-вопросы:**")
cols = st.columns(len(DEMO_QUESTIONS))
for i, (col, q) in enumerate(zip(cols, DEMO_QUESTIONS)):
    with col:
        if st.button(q[:50] + "..." if len(q) > 50 else q, key=f"demo_{i}"):
            st.session_state["selected_query"] = q

# ── Ввод вопроса ───────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Введите вопрос:",
        value=st.session_state.get("selected_query", ""),
        placeholder="Например: какие задачи цифровой трансформации в здравоохранении?",
    )

with col2:
    top_k = st.selectbox("Топ чанков", [3, 5, 10], index=0)

# ── Поиск и генерация ──────────────────────────────────────────────────────────

if query:
    with st.spinner("Ищу релевантные фрагменты..."):
        chunks = retriever.search(query, top_k=top_k)

    # Найденные фрагменты
    st.subheader("📄 Найденные фрагменты")
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0)
        region = chunk.get("region", chunk.get("doc_id", "?"))
        page = chunk.get("page", "?")

        color = "🟢" if score > 0.1 else "🟡" if score > 0.05 else "🔴"
        with st.expander(
            f"{color} #{i} | {region} — стр. {page} | score: {score:.4f}",
            expanded=(i == 1),
        ):
            st.markdown(chunk["text"][:800])

    st.divider()

    # Генерация ответа
    st.subheader("💬 Ответ ассистента")

    has_key = bool(os.getenv("OPENROUTER_API_KEY", ""))

    if not has_key:
        st.warning(
            "⚠️ OPENROUTER_API_KEY не задан. Создай файл `.env` с ключом.\n\n"
            "Показываю только найденные фрагменты без генерации ответа."
        )
    else:
        with st.spinner("Генерирую ответ..."):
            from app.generator import generate_answer
            result = generate_answer(query, chunks)

        if result["refused"]:
            st.warning(f"🚫 {result['answer']}")
        else:
            st.markdown(result["answer"])

            if result["sources"]:
                st.markdown("**Источники:**")
                for s in result["sources"]:
                    st.markdown(
                        f"- 📍 **{s['region']}**, стр. {s['page']} "
                        f"_(релевантность: {s['score']})_"
                    )
