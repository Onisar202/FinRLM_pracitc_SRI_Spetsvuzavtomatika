import sys
import io
import pathlib
import contextlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
from src.finrlm.rlm.recursive import run_rlm, get_last_sources

st.set_page_config(page_title="FinRLM", page_icon="📈")

with st.sidebar:
    st.header("FinRLM")
    st.caption(
        "Гибридная Q&A система по финансовым новостям.\n\n"
        "**RAG** — гибридный поиск (BGE-M3 dense+sparse, RRF) + "
        "реранжирование (BGE-reranker) + генерация ответа с цитатами.\n\n"
        "**RLM** — рекурсивно вызывает `rag_search` для подвопросов "
        "и синтезирует итоговый ответ."
    )
    st.divider()
    st.caption("Требуется запущенный Qdrant (:6333) и vLLM (:8000).")

st.title("📈 FinRLM")
st.caption("Вопросно-ответная система по финансовым новостям (RAG + RLM)")

query = st.text_input(
    "Вопрос",
    placeholder="Например: Сбербанк новости  ·  Сравни дивиденды Газпрома и Норникеля",
)

if st.button("Спросить", type="primary") and query.strip():
    trace = io.StringIO()
    answer, error = None, None
    with st.spinner("RLM рассуждает и ищет по новостям…"):
        with contextlib.redirect_stdout(trace):
            try:
                answer = run_rlm(query.strip())
            except Exception as e:
                error = e
        sources = get_last_sources()

    if error is not None:
        st.error(
            f"RLM не успел завершить рассуждение ({type(error).__name__}). "
            "Попробуйте переформулировать вопрос короче или повторить запрос."
        )
    else:
        st.subheader("Ответ")
        st.write(answer)

        if sources:
            st.subheader("Источники")
            for i, s in enumerate(sources, 1):
                st.markdown(
                    f"{i}. [{s['title']}]({s['url']})  \n"
                    f"<sub>{s['source']} · {s['date']}</sub>",
                    unsafe_allow_html=True,
                )

    with st.expander("Показать reasoning RLM"):
        st.text(trace.getvalue())
