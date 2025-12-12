# nutritionist_agent.py

from typing import Optional
from time import sleep

from mistralai import Mistral
from langchain_core.language_models import BaseLanguageModel
from langchain_community.retrievers import ArxivRetriever


def nutritionist_agent(
    llm: BaseLanguageModel,
    user_prompt: str,
    retriever: Optional[object] = None
) -> str:
    """
    Агент-нутрициолог. Формирует чёткий план питания.
    Если передан retriever (например, ArxivRetriever), делает второй запрос,
    уточняя план питания с учётом релевантной научной информации.

    Аргументы:
        llm: экземпляр LLM, совместимый с .chat(prompt)
        user_prompt: строка запроса пользователя
        retriever: опциональный retrивер (например, ArxivRetriever)

    Возвращает:
        Строка с финальным планом питания.
    """

    system_prompt = """
        Ты агент-нутрициолог. Твоя задача — составить чёткий, структурированный
        план питания по приёмам пищи (завтрак, обед, ужин, перекусы).
        Не добавляй лишних рекомендаций или тренировочных планов.
        Отвечай только планом питания.
    """

    # 1. Первый запрос LLM — базовый план
    first_prompt = f"{system_prompt}\n\nЗапрос пользователя:\n{user_prompt}"
    first_response = llm.chat(first_prompt)

    sleep(1)

    # 2. Проверяем наличие retriever
    if retriever:
        relevant_docs = retriever.invoke(user_prompt)
        additional_info = "\n".join([doc.page_content for doc in relevant_docs])

        sleep(1)

        second_prompt = f"""
            У меня есть план питания, а также некоторая научная информация (например, выдержки из статей).
            Тебе нужно обновить план питания, добавив полезные уточнения или микро-правки,
            основанные на этих данных. Но важно: ответ должен остаться именно планом питания,
            без пояснений, цитат, списков статей и прочего.
            
            Вот дополнительная информация:
            {additional_info}
            
            Вот базовый план питания:
            {first_response}
            
            Отправь только обновлённый план питания.
        """

        final_response = llm.chat(second_prompt)
    else:
        final_response = first_response

    return final_response
