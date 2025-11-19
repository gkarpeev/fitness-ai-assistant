"""Coordinator for managing multiple fitness agents."""
from typing import Dict, Any, Optional
from langchain_anthropic import ChatAnthropic

import config
from src.agents.workout_agent import WorkoutPlanningAgent
from src.agents.nutrition_agent import NutritionVideoAgent
from src.tools.custom_tools import FitnessTools
from src.rag.vectorstore import FitnessVectorStore
from src.models.schemas import UserProfile


class FitnessCoordinator:
    """Coordinates between workout and nutrition agents."""
    
    def __init__(self, vectorstore: FitnessVectorStore):
        """Initialize the coordinator with both agents."""
        self.vectorstore = vectorstore
        
        # Initialize tools
        self.fitness_tools = FitnessTools(vectorstore)
        tools = self.fitness_tools.create_langchain_tools()
        
        # Initialize agents
        self.workout_agent = WorkoutPlanningAgent(tools)
        self.nutrition_agent = NutritionVideoAgent(tools)
        
        # Initialize routing LLM
        self.router_llm = ChatAnthropic(
            api_key=config.ANTHROPIC_API_KEY,
            model=config.CLAUDE_MODEL,
            temperature=0.3
        )
        
        print("✅ Fitness Coordinator initialized with both agents")
    
    def _classify_query(self, user_message: str) -> str:
        """Classify which agent should handle the query."""
        classification_prompt = f"""
        Проанализируй запрос пользователя и определи, какой агент должен его обработать:
        
        - "workout": если запрос о тренировках, упражнениях, программе тренировок, технике выполнения
        - "nutrition": если запрос о питании, добавках, диете, калориях, макронутриентах
        - "both": если запрос требует информации от обоих агентов (например, полный план набора массы)
        
        Запрос пользователя: "{user_message}"
        
        Ответь ТОЛЬКО одним словом: workout, nutrition или both
        """
        
        try:
            response = self.router_llm.invoke(classification_prompt)
            classification = response.content.strip().lower()
            
            if classification not in ["workout", "nutrition", "both"]:
                # Default to workout if unclear
                return "workout"
            
            return classification
        except Exception as e:
            print(f"Error in classification: {e}")
            return "workout"
    
    def register_user(self, user_profile: UserProfile):
        """Register a user profile."""
        self.fitness_tools.user_profiles[user_profile.user_id] = user_profile
        print(f"✅ User profile registered: {user_profile.user_id}")
    
    def process_query(
        self,
        user_message: str,
        user_id: Optional[str] = None,
        force_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user query and route to appropriate agent(s).
        
        Args:
            user_message: The user's question or request
            user_id: Optional user ID for personalization
            force_agent: Force routing to specific agent ('workout', 'nutrition', or 'both')
        
        Returns:
            Dictionary with response and metadata
        """
        # Get user context if available
        user_context = None
        if user_id and user_id in self.fitness_tools.user_profiles:
            profile = self.fitness_tools.user_profiles[user_id]
            user_context = {
                "user_id": user_id,
                "goal": profile.goal.value,
                "experience_level": profile.experience_level.value,
                "equipment_access": profile.equipment_access.value,
                "injuries": profile.injuries,
                "dietary_restrictions": profile.dietary_restrictions
            }
        
        # Classify query
        if force_agent:
            agent_type = force_agent
        else:
            agent_type = self._classify_query(user_message)
        
        print(f"\n{'='*60}")
        print(f"🎯 Routing to: {agent_type.upper()}")
        print(f"{'='*60}\n")
        
        # Route to appropriate agent(s)
        result = {
            "agent_type": agent_type,
            "user_message": user_message,
            "response": ""
        }
        
        try:
            if agent_type == "workout":
                response = self.workout_agent.process(user_message, user_context)
                result["response"] = response
                
            elif agent_type == "nutrition":
                response = self.nutrition_agent.process(user_message, user_context)
                result["response"] = response
                
            elif agent_type == "both":
                # Get responses from both agents
                workout_response = self.workout_agent.process(
                    f"Создай программу тренировок для: {user_message}",
                    user_context
                )
                
                nutrition_response = self.nutrition_agent.process(
                    f"Создай план питания и подбери добавки для: {user_message}",
                    user_context
                )
                
                # Combine responses
                combined = f"""
# КОМПЛЕКСНЫЙ ПЛАН ДОСТИЖЕНИЯ ЦЕЛИ

## 🏋️ ПРОГРАММА ТРЕНИРОВОК

{workout_response}

---

## 🥗 ПЛАН ПИТАНИЯ И ДОБАВКИ

{nutrition_response}

---

💡 **Важно**: Успех зависит от комплексного подхода - тренировки, питание и восстановление одинаково важны!
                """
                
                result["response"] = combined
                result["workout_response"] = workout_response
                result["nutrition_response"] = nutrition_response
        
        except Exception as e:
            result["response"] = f"Произошла ошибка: {str(e)}"
            result["error"] = str(e)
        
        return result
    
    async def aprocess_query(
        self,
        user_message: str,
        user_id: Optional[str] = None,
        force_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of process_query."""
        user_context = None
        if user_id and user_id in self.fitness_tools.user_profiles:
            profile = self.fitness_tools.user_profiles[user_id]
            user_context = {
                "user_id": user_id,
                "goal": profile.goal.value,
                "experience_level": profile.experience_level.value,
                "equipment_access": profile.equipment_access.value,
                "injuries": profile.injuries,
                "dietary_restrictions": profile.dietary_restrictions
            }
        
        if force_agent:
            agent_type = force_agent
        else:
            agent_type = self._classify_query(user_message)
        
        result = {
            "agent_type": agent_type,
            "user_message": user_message,
            "response": ""
        }
        
        try:
            if agent_type == "workout":
                response = await self.workout_agent.aprocess(user_message, user_context)
                result["response"] = response
                
            elif agent_type == "nutrition":
                response = await self.nutrition_agent.aprocess(user_message, user_context)
                result["response"] = response
                
            elif agent_type == "both":
                import asyncio
                
                # Run both agents concurrently
                workout_task = self.workout_agent.aprocess(
                    f"Создай программу тренировок для: {user_message}",
                    user_context
                )
                nutrition_task = self.nutrition_agent.aprocess(
                    f"Создай план питания и подбери добавки для: {user_message}",
                    user_context
                )
                
                workout_response, nutrition_response = await asyncio.gather(
                    workout_task, nutrition_task
                )
                
                combined = f"""
# КОМПЛЕКСНЫЙ ПЛАН ДОСТИЖЕНИЯ ЦЕЛИ

## 🏋️ ПРОГРАММА ТРЕНИРОВОК

{workout_response}

---

## 🥗 ПЛАН ПИТАНИЯ И ДОБАВКИ

{nutrition_response}

---

💡 **Важно**: Успех зависит от комплексного подхода - тренировки, питание и восстановление одинаково важны!
                """
                
                result["response"] = combined
                result["workout_response"] = workout_response
                result["nutrition_response"] = nutrition_response
        
        except Exception as e:
            result["response"] = f"Произошла ошибка: {str(e)}"
            result["error"] = str(e)
        
        return result