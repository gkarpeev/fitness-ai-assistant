"""Custom tools for fitness AI agents."""
from typing import Dict, List, Optional
from langchain.tools import Tool
from pydantic import BaseModel, Field

from src.rag.vectorstore import FitnessVectorStore
from src.models.schemas import UserProfile


class ExerciseSearchInput(BaseModel):
    """Input for exercise search."""
    query: str = Field(description="Search query for exercises")
    muscle_groups: Optional[List[str]] = Field(
        default=None,
        description="Filter by muscle groups"
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="Filter by difficulty: beginner, intermediate, advanced"
    )


class SupplementSearchInput(BaseModel):
    """Input for supplement search."""
    query: str = Field(description="Search query for supplements")
    goal: Optional[str] = Field(
        default=None,
        description="User's fitness goal"
    )


class NutritionSearchInput(BaseModel):
    """Input for nutrition knowledge search."""
    query: str = Field(description="Search query for nutrition information")


class FitnessTools:
    """Collection of tools for fitness agents."""
    
    def __init__(self, vectorstore: FitnessVectorStore):
        self.vectorstore = vectorstore
        self.user_profiles: Dict[str, UserProfile] = {}
    
    def search_exercises_tool(self, query: str, **kwargs) -> str:
        """Search for exercises in the knowledge base."""
        try:
            results = self.vectorstore.search_exercises(
                query=query,
                n_results=5
            )
            
            if not results:
                return "Упражнения не найдены. Попробуйте изменить запрос."
            
            output = "Найденные упражнения:\n\n"
            for i, result in enumerate(results, 1):
                meta = result["metadata"]
                output += f"{i}. {meta['name']}\n"
                output += f"   Мышцы: {meta['muscle_groups']}\n"
                output += f"   Оборудование: {meta['equipment']}\n"
                output += f"   Сложность: {meta['difficulty']}\n"
                output += f"   Описание: {meta['description'][:200]}...\n\n"
            
            return output
        except Exception as e:
            return f"Ошибка поиска упражнений: {str(e)}"
    
    def search_supplements_tool(self, query: str) -> str:
        """Search for supplements in the knowledge base."""
        try:
            results = self.vectorstore.search_supplements(
                query=query,
                n_results=5
            )
            
            if not results:
                return "Добавки не найдены. Попробуйте изменить запрос."
            
            output = "Найденные добавки:\n\n"
            for i, result in enumerate(results, 1):
                meta = result["metadata"]
                output += f"{i}. {meta['name']}\n"
                output += f"   Категория: {meta['category']}\n"
                output += f"   Дозировка: {meta['dosage']}\n"
                output += f"   Уровень доказательств: {meta['evidence_level']}\n"
                output += f"   Время приема: {meta['timing']}\n"
                output += f"   Заметки: {meta.get('notes', 'Нет')[:200]}...\n\n"
            
            return output
        except Exception as e:
            return f"Ошибка поиска добавок: {str(e)}"
    
    def search_nutrition_tool(self, query: str) -> str:
        """Search for nutrition information."""
        try:
            results = self.vectorstore.search_nutrition(
                query=query,
                n_results=3
            )
            
            if not results:
                return "Информация о питании не найдена. Попробуйте изменить запрос."
            
            output = "Найденная информация о питании:\n\n"
            for i, result in enumerate(results, 1):
                meta = result["metadata"]
                output += f"{i}. {meta['topic']}\n"
                output += f"{meta['content'][:500]}...\n\n"
            
            return output
        except Exception as e:
            return f"Ошибка поиска информации о питании: {str(e)}"
    
    def get_user_profile_tool(self, user_id: str) -> str:
        """Get user profile information."""
        if user_id not in self.user_profiles:
            return "Профиль пользователя не найден. Пожалуйста, создайте профиль сначала."
        
        profile = self.user_profiles[user_id]
        output = f"""
        Профиль пользователя {user_id}:
        - Цель: {profile.goal.value}
        - Уровень: {profile.experience_level.value}
        - Оборудование: {profile.equipment_access.value}
        - Травмы/ограничения: {', '.join(profile.injuries) if profile.injuries else 'Нет'}
        - Диетические ограничения: {', '.join(profile.dietary_restrictions) if profile.dietary_restrictions else 'Нет'}
        """
        
        if profile.age:
            output += f"\n- Возраст: {profile.age}"
        if profile.weight:
            output += f"\n- Вес: {profile.weight} кг"
        if profile.height:
            output += f"\n- Рост: {profile.height} см"
        
        return output.strip()
    
    def calculate_calories_tool(self, weight: float, height: float, age: int, gender: str, activity: str, goal: str) -> str:
        """Calculate daily calorie needs."""
        try:
            # BMR calculation (Mifflin-St Jeor)
            if gender.lower() in ["male", "мужской", "м"]:
                bmr = 10 * weight + 6.25 * height - 5 * age + 5
            else:
                bmr = 10 * weight + 6.25 * height - 5 * age - 161
            
            # Activity multipliers
            activity_multipliers = {
                "sedentary": 1.2,
                "light": 1.375,
                "moderate": 1.55,
                "active": 1.725,
                "very_active": 1.9
            }
            
            multiplier = activity_multipliers.get(activity.lower(), 1.55)
            tdee = bmr * multiplier
            
            # Adjust for goal
            if "loss" in goal.lower() or "похудение" in goal.lower():
                target = tdee - 500
                protein = weight * 2.2
                note = "дефицит 500 ккал для похудения"
            elif "gain" in goal.lower() or "набор" in goal.lower():
                target = tdee + 400
                protein = weight * 2.0
                note = "профицит 400 ккал для набора массы"
            else:
                target = tdee
                protein = weight * 1.8
                note = "поддержание веса"
            
            output = f"""
            Расчет калорий:
            - BMR (базовый обмен): {bmr:.0f} ккал
            - TDEE (общий расход): {tdee:.0f} ккал
            - Целевые калории: {target:.0f} ккал ({note})
            
            Рекомендуемые макронутриенты:
            - Белок: {protein:.0f}г ({protein*4:.0f} ккал)
            - Жиры: {weight * 1.0:.0f}г ({weight * 1.0 * 9:.0f} ккал)
            - Углеводы: {(target - protein*4 - weight*9) / 4:.0f}г
            """
            
            return output.strip()
        except Exception as e:
            return f"Ошибка расчета калорий: {str(e)}"
    
    def create_langchain_tools(self) -> List[Tool]:
        """Create LangChain Tool objects."""
        return [
            Tool(
                name="search_exercises",
                func=self.search_exercises_tool,
                description="Ищет упражнения в базе знаний. Используй для подбора упражнений по мышечным группам, оборудованию или целям. Входные данные: текстовый запрос с описанием нужных упражнений."
            ),
            Tool(
                name="search_supplements",
                func=self.search_supplements_tool,
                description="Ищет спортивные добавки в базе знаний. Используй для получения информации о добавках, их эффективности, дозировках и противопоказаниях. Входные данные: название добавки или тип (например, 'для набора массы', 'энергия')."
            ),
            Tool(
                name="search_nutrition",
                func=self.search_nutrition_tool,
                description="Ищет информацию о питании в базе знаний. Используй для получения рекомендаций по питанию, макронутриентам, времени приема пищи. Входные данные: вопрос о питании."
            ),
            Tool(
                name="get_user_profile",
                func=self.get_user_profile_tool,
                description="Получает профиль пользователя с его целями, уровнем подготовки, ограничениями. Входные данные: user_id."
            ),
            Tool(
                name="calculate_calories",
                func=lambda x: self.calculate_calories_tool(**eval(x)),
                description="Рассчитывает дневную норму калорий и макронутриентов. Входные данные: словарь с ключами weight, height, age, gender, activity, goal (в формате строки словаря Python)."
            )
        ]