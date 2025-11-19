"""Configuration settings for the AI Fitness Assistant."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"))

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Model settings
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))

# ChromaDB collections
COLLECTIONS = {
    "exercises": "fitness_exercises",
    "supplements": "fitness_supplements",
    "videos": "fitness_videos",
    "nutrition": "fitness_nutrition"
}

# Agent configurations
WORKOUT_AGENT_NAME = "Workout Planning Agent"
NUTRITION_AGENT_NAME = "Nutrition & Video Agent"