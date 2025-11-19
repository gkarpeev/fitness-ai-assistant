"""Data models and schemas for the fitness assistant."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class FitnessGoal(str, Enum):
    """User fitness goals."""
    MUSCLE_GAIN = "muscle_gain"
    WEIGHT_LOSS = "weight_loss"
    STRENGTH = "strength"
    ENDURANCE = "endurance"
    GENERAL_FITNESS = "general_fitness"


class ExperienceLevel(str, Enum):
    """User experience level."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class EquipmentAccess(str, Enum):
    """Available equipment."""
    HOME_MINIMAL = "home_minimal"
    HOME_EQUIPPED = "home_equipped"
    GYM = "gym"


class UserProfile(BaseModel):
    """User profile information."""
    user_id: str
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    goal: FitnessGoal
    experience_level: ExperienceLevel
    equipment_access: EquipmentAccess
    injuries: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    dietary_restrictions: List[str] = Field(default_factory=list)


class Exercise(BaseModel):
    """Exercise information."""
    name: str
    muscle_groups: List[str]
    equipment: List[str]
    difficulty: str
    description: str
    technique_points: List[str]
    video_url: Optional[str] = None


class Supplement(BaseModel):
    """Supplement information."""
    name: str
    category: str
    benefits: List[str]
    dosage: str
    timing: str
    contraindications: List[str]
    evidence_level: str  # "high", "medium", "low"


class WorkoutPlan(BaseModel):
    """Generated workout plan."""
    user_id: str
    goal: str
    duration_weeks: int
    days_per_week: int
    exercises: List[Dict[str, Any]]
    notes: str


class NutritionAdvice(BaseModel):
    """Nutrition and supplement advice."""
    user_id: str
    goal: str
    recommended_supplements: List[Dict[str, Any]]
    nutrition_guidelines: Dict[str, Any]
    videos: List[Dict[str, str]]