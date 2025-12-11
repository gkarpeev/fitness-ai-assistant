# manager_agent.py

import re
from typing import List, Optional
from time import sleep


def manager_agent(
    llm,
    user_query: str,
    trainer_agent_fn,
    nutritionist_agent_fn,
    coach_retriever=None,
    nutritionist_retriever=None
) -> str:
    """
    Агент-менеджер. Определяет, как обработать запрос пользователя:
    0 — не по теме,
    1 — отправить агенту тренеру,
    2 — отправить агенту нутрициологу.

    Аргументы:
        llm: LLM с методом .chat(prompt)
        user_query: запрос пользователя
        trainer_agent_fn: функция агента тренера (llm, user_query, retriever=None)
        nutritionist_agent_fn: функция нутрициолога (llm, user_query, retriever=None)
        coach_retriever: опциональный retriever для тренера
        nutritionist_retriever: опциональный retriever для нутрициолога

    Возвращает:
        Строку с финальным ответом.
    """

    system_prompt = f"""
Ты агент-менеджер. У тебя есть три опции:
0) Ответить, что запрос не относится к тренировкам или питанию.
1) Отдать запрос агенту тренеру.
2) Отдать запрос агенту нутрициологу.

Ты должен вернуть список опций: [1], [2], [1,2] или [0].
Никакого текста кроме списка.

Запрос пользователя: {user_query}
    """

    routing_response = llm.chat(system_prompt)

    # Извлекаем список опций, которые определил LLM
    match = re.search(r"\[.*?\]", routing_response)
    try:
        selected_options: List[int] = eval(match.group()) if match else []
        if not isinstance(selected_options, list):
            selected_options = []
    except:
        selected_options = []

    responses = []

    sleep(1)
    # Агент тренер
    if 1 in selected_options and trainer_agent_fn:
        trainer_answer = trainer_agent_fn(
            llm,
            user_query,
            retriever=coach_retriever
        )
        responses.append(f"План тренировок:\n{trainer_answer}")

    sleep(1)

    # Агент нутрициолог
    if 2 in selected_options and nutritionist_agent_fn:
        nutrition_answer = nutritionist_agent_fn(
            llm,
            user_query,
            retriever=nutritionist_retriever
        )
        responses.append(f"План питания:\n{nutrition_answer}")

    # Ни один агент не был выбран
    if not responses:
        return "Извините, я могу помочь только с тренировками и питанием."

    return "\n\n".join(responses)
