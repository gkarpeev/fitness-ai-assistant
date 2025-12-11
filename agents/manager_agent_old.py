# manager_agent.py
import os
import re
from mistralai import Mistral
# from langchain.chat_models import MistralChat
# from langchain.chains import LLMChain, RetrievalQA
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.vectorstores import FAISS
# from langchain_community.retrievers import ArxivRetriever
# from langchain.embeddings import HuggingFaceEmbeddings

# -------------------------
# 1. Инициализация LLM
# -------------------------
# llm = MistralChat(model_name="mistral-large", temperature=0)

# -------------------------
# 2. Инициализация агентов
# -------------------------
# Тренер (FAISS база)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER_DB_PATH = os.path.join(PROJECT_ROOT, "db/trainer_vectordb")


# -------------------------
# 3. Промпт менеджера
# -------------------------
def user_message_manager(inquiry):
    user_message = (
        f"""Ты агент-менеджер. У тебя есть три опции для обработки запроса пользователя:
        0) Если запрос не по нашей тематике (тренировки или питание) — скажи пользователю, что помочь не можешь.
        1) Отправить запрос агенту тренеру, если нужен план тренировок.
        2) Отправить запрос агенту нутрициологу, если нужен план питания.

        Твоя задача: проанализировать запрос и вернуть список опций, которые нужно выполнить (например: [1], [2], [1,2] или [0]).
        Возвращай строго в формате списка чисел.

        Запрос пользователя: {inquiry}
        """
    )
    return user_message

# system_prompt = SystemMessagePromptTemplate.from_template(system_template)
# human_prompt = HumanMessagePromptTemplate.from_template("{user_query}")
# prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# manager_chain = LLMChain(llm=llm, prompt=prompt)

# -------------------------
# 4. Логика агента менеджера
# -------------------------
def manager_agent(user_query: str) -> str:
    # 1. Определяем опции через LLM
    selected_options_str = manager_chain.run(user_query)
    
    # Преобразуем строку LLM в список чисел
    try:
        selected_options = eval(selected_options_str)
        if not isinstance(selected_options, list):
            selected_options = []
    except:
        selected_options = []

    responses = []

    # 2. Вызов нужных агентов
    if 1 in selected_options:
        trainer_response = trainer_qa.run(user_query)
        responses.append(f"План тренировок:\n{trainer_response}")

    if 2 in selected_options:
        nutrition_response = nutrition_qa.run(user_query)
        responses.append(f"План питания:\n{nutrition_response}")

    # 3. Если ни один агент не выбран (0)
    if not responses:
        return "Извините, я могу помочь только с планом тренировок и схемой питания."

    # 4. Склеиваем ответы
    return "\n\n".join(responses)



def run_mistral(messages, user_format=True, model="mistral-medium-latest"):
    api_key = "dNLiGfHEHQVIFTY1t0gAecNAljgBsnBf"
    client = Mistral(api_key=api_key)
    if user_format:
        messages = [
            {"role":"user", "content":messages}
        ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response

# -------------------------
# 5. Тестирование через __main__
# -------------------------
if __name__ == "__main__":
    # print("=== Менеджер Агент ===")
    # while True:
    #     user_input = input("\nВведите запрос (или 'exit' для выхода): ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         break
    #     answer = manager_agent(user_input)
    #     print("\n--- Ответ менеджера ---\n")
    #     print(answer)
    some_prompt = "Привет, я Егор, мне 20 лет, я вешу 74 килограмма, составь мне план тренировок для набора мышечной массы для игры в хоккей, а также план питания."
    model_answer = run_mistral(user_message_manager(some_prompt))
    print(model_answer.choices[0].message.content)
    match = re.search(r"\[.*?\]", model_answer.choices[0].message.content)
    try:
        selected_options = eval(match.group())
        if not isinstance(selected_options, list):
            selected_options = []
    except:
        selected_options = []
    print("bebra")
    print(selected_options)
    print(type(selected_options))
