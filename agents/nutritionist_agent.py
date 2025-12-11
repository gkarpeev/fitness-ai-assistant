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

    # print(f"\n Первый ответ нутрициолога:\n{first_response}\n")
    sleep(1)

    # 2. Проверяем наличие retriever
    if retriever:
        relevant_docs = retriever.invoke(user_prompt)
        additional_info = "\n".join([doc.page_content for doc in relevant_docs])

        # print(f"\nДополнительная информация из retriever:\n{additional_info}\n")
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


# def run_mistral_client(model="mistral-medium-latest"):
#     api_key = "dNLiGfHEHQVIFTY1t0gAecNAljgBsnBf"
#     return Mistral(api_key=api_key), model


# class SimpleLLM:
#     """
#     Унифицированный класс-обёртка под Mistral API,
#     чтобы иметь единый интерфейс .chat(prompt).
#     """
#     def __init__(self, client, model):
#         self.client = client
#         self.model = model

#     def chat(self, prompt: str) -> str:
#         response = self.client.chat.complete(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content


# if __name__ == "__main__":
#     # Пример теста
#     test_prompt = "Привет, я Егор, мне 20 лет, я вешу 74 килограмма, составь мне план тренировок для набора мышечной массы для игры в хоккей, а также план питания."

#     mistral_client, mistral_model = run_mistral_client()
#     llm = SimpleLLM(mistral_client, mistral_model)

#     retriever = ArxivRetriever(load_max_docs=2)
#     print("\n=== Ответ нутрициолога ===\n")
#     result = nutritionist_agent(llm, test_prompt, retriever=None)
#     print(result)
