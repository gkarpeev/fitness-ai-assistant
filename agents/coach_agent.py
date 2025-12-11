import os
from typing import Optional

from mistralai import Mistral
from time import sleep

from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



def trainer_agent(llm: BaseLanguageModel, user_prompt: str, retriever: Optional[object] = None) -> str:
    """
    Агент-тренер. Составляет план тренировок по дням недели.
    Если предоставлен retriever, использует релевантные документы для уточнения плана.
    
    Аргументы:
        llm: экземпляр LLM (например, Mistral через LangChain)
        user_prompt: строка запроса пользователя
        retriever: опциональный retriever (FAISS) для поиска дополнительной информации
        
    Возвращает:
        Итоговый текст с планом тренировок
    """
    
    # 1. Системный промпт
    system_prompt = """
        Ты агент-тренер. Составь четкий план тренировок на неделю для стандартного тренажерного зала.
        Каждый тренировочный день указывай упражнения, количество подходов и повторений. Не пиши ничего кроме плана тренировок - твоя задача написать
        только план тренировок, план питания ты писать не должен!!!

        Вот пример плана:

        День 1 – Грудь, спина, плечи:
        - Жим штанги лёжа – 3 подхода по 10 повторений
        - Тяга верхнего блока к груди – 3 подхода по 10 повторений
        - Жим гантелей сидя (плечи) – 3 подхода по 12 повторений
        - Разводка гантелей лёжа – 2 подхода по 12 повторений

        День 2 – Ноги и нижняя часть тела:
        - Приседания со штангой – 4 подхода по 10 повторений
        - Жим ногами – 3 подхода по 12 повторений
        - Выпады с гантелями – 3 подхода по 12 повторений
        - Подъёмы на носки – 3 подхода по 15 повторений

        День 3 – Руки и кора:
        - Подтягивания или тяга горизонтального блока – 3 подхода до отказа
        - Сгибания рук со штангой – 3 подхода по 12 повторений
        - Французский жим – 3 подхода по 12 повторений
        - Планка – 3 подхода по 30–60 секунд

        Рекомендации:
        - Между тренировками делать день отдыха (например: понедельник, среда, пятница)
        - Разминка 5–10 минут перед каждой тренировкой обязательна
        - Для набора мышечной массы постепенно увеличивать вес снарядов
        """
    
    # 2. Первый запрос к LLM
    first_prompt = f"{system_prompt}\n\nЗапрос пользователя:\n{user_prompt}"
    first_response = llm.chat(first_prompt)

    # print(f"\n Первый ответ\n {first_response}\n")
    sleep(1)
    
    # 3. Второй запрос с информацией из retriever
    if retriever:
        relevant_docs = retriever.invoke(user_prompt)
        additional_info = "\n".join([doc.page_content for doc in relevant_docs])

        # print(f"\n\n\nadditional_info: \n {additional_info} \n\n\n")
        second_prompt = f"""
            У меня есть план тренировок в тренажерном зале для спортсмена, а также дополнительная информация - в которой есть
            ссылки на you-tube видео - иллюстрирующие какие-то упражнения.
            Тебе надо отправить мне новый план, сделанный на основе прежнего, в который ты максимально постараешься добавить
            иллюстрации к упражнениям из дополнительной информации (просто добавь ссылки на you-tube ролики под 
            соответствующее упражнение, где это возможно). Обязательно добавь хотябы 2 ссылки, по-возможности как можно больше.

            Вот дополнительная информация:\n {additional_info} \n\n
            Вот прежний план тренировок: \n {first_response} \n\n
            Отправь только проиллюстрированный план.
            """
        final_response = llm.chat(second_prompt)
    else:
        final_response = first_response
    
    return final_response


# def run_mistral_client(model="mistral-medium-latest"):
#     api_key = "dNLiGfHEHQVIFTY1t0gAecNAljgBsnBf"
#     return Mistral(api_key=api_key), model

# class SimpleLLM:
#     def __init__(self, client, model):
#         self.client = client
#         self.model = model

#     def chat(self, prompt: str) -> str:
#         response = self.client.chat.complete(
#             model=self.model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response.choices[0].message.content

# if __name__ == "__main__":
#     some_prompt = "Привет, я Егор, мне 20 лет, я вешу 74 килограмма, составь мне план тренировок для набора мышечной массы для игры в хоккей, а также план питания."

#     CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#     DB_PATH = os.path.join(CURRENT_DIR, "..", "db", "trainer_vectordb")

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     vectordb = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
#     retriever = vectordb.as_retriever(k=15)

#     mistral_client, mistral_model = run_mistral_client()

#     llm = SimpleLLM(mistral_client, mistral_model)

#     print("\n=== Ответ агента тренера ===\n")
#     result = trainer_agent(llm, some_prompt, retriever=retriever)
#     print(result)