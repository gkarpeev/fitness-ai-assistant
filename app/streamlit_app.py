import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st

from time import sleep
from mistralai import Mistral
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import ArxivRetriever

# Импортируем агентов
from agents.coach_agent import trainer_agent
from agents.nutritionist_agent import nutritionist_agent
from agents.manager_agent import manager_agent


###########################################################################
# 1. Настройки Streamlit
###########################################################################

st.set_page_config(
    page_title="AI Фитнес-коуч",
    page_icon="🏋️",
    layout="centered"
)

st.title("AI Фитнес-коуч")
st.write("""
Введите запрос: ваш возраст, вес, цели, предпочтения, ограничения, режим тренировки или питания.
Примеры:
- "Мне 20 лет, хочу план тренировок для набора мышечной массы"
- "Подбери мне питание для похудения"
- "Сделай тренировки для хоккеиста + питание"
""")


###########################################################################
# 2. Инициализация объектов один раз
###########################################################################

def init_objects():

    # Инициализация LLM
    api_key = "dNLiGfHEHQVIFTY1t0gAecNAljgBsnBf"
    client = Mistral(api_key=api_key)
    model = "mistral-medium-latest"

    class SimpleLLM:
        def __init__(self, client, model):
            self.client = client
            self.model = model

        def chat(self, prompt: str) -> str:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

    llm = SimpleLLM(client, model)

    # Векторная база тренера
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(CURRENT_DIR, "..", "db", "trainer_vectordb")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    coach_retriever = vectordb.as_retriever(k=15)

    # Ретривер нутрициолога (ArXiv)
    nutritionist_retriever = ArxivRetriever(load_max_docs=5)

    return llm, coach_retriever, nutritionist_retriever


# Создаем объекты только один раз в сессии
if "initialized" not in st.session_state:

    (
        st.session_state.llm,
        st.session_state.coach_retriever,
        st.session_state.nutritionist_retriever
    ) = init_objects()

    st.session_state.initialized = True


###########################################################################
# 3. UI для ввода
###########################################################################

user_query = st.text_area("Ваш запрос:")

if st.button("Сгенерировать программу") and user_query.strip():

    llm = st.session_state.llm
    coach_ret = st.session_state.coach_retriever
    nutr_ret = st.session_state.nutritionist_retriever

    with st.spinner("Генерирую ответ..."):

        final_answer = manager_agent(
            llm=llm,
            user_query=user_query,
            trainer_agent_fn=trainer_agent,
            nutritionist_agent_fn=nutritionist_agent,
            coach_retriever=coach_ret,
            nutritionist_retriever=nutr_ret
        )

    st.write("### Результат:")
    st.write(final_answer)
