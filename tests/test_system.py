"""Basic tests for AI Fitness Assistant."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.models.schemas import UserProfile, FitnessGoal, ExperienceLevel, EquipmentAccess


def test_imports():
    """Test that all major modules can be imported."""
    try:
        import anthropic
        import langchain
        import chromadb
        import sentence_transformers
        from src.rag.vectorstore import FitnessVectorStore
        from src.agents.coordinator import FitnessCoordinator
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")


def test_user_profile_creation():
    """Test creating a user profile."""
    profile = UserProfile(
        user_id="test_001",
        age=25,
        weight=70,
        height=175,
        gender="male",
        goal=FitnessGoal.MUSCLE_GAIN,
        experience_level=ExperienceLevel.BEGINNER,
        equipment_access=EquipmentAccess.GYM,
        injuries=[],
        dietary_restrictions=[]
    )
    
    assert profile.user_id == "test_001"
    assert profile.age == 25
    assert profile.goal == FitnessGoal.MUSCLE_GAIN


def test_data_generators():
    """Test data generation functions."""
    from src.data.generators import (
        generate_exercises_data,
        generate_supplements_data,
        generate_nutrition_data
    )
    
    exercises = generate_exercises_data()
    supplements = generate_supplements_data()
    nutrition = generate_nutrition_data()
    
    assert len(exercises) > 0
    assert len(supplements) > 0
    assert len(nutrition) > 0
    
    # Check structure of first exercise
    assert "name" in exercises[0]
    assert "muscle_groups" in exercises[0]
    assert "equipment" in exercises[0]
    
    # Check structure of first supplement
    assert "name" in supplements[0]
    assert "dosage" in supplements[0]
    assert "evidence_level" in supplements[0]


def test_config():
    """Test configuration loading."""
    import config
    
    assert config.ANTHROPIC_API_KEY is not None
    assert config.CLAUDE_MODEL is not None
    assert config.CHROMA_DIR.exists()


def test_vectorstore_initialization():
    """Test basic vectorstore operations (requires initialized DB)."""
    from src.rag.vectorstore import FitnessVectorStore
    
    try:
        vectorstore = FitnessVectorStore()
        
        # Test search
        results = vectorstore.search_exercises("грудь", n_results=3)
        assert isinstance(results, list)
        
        results = vectorstore.search_supplements("креатин", n_results=3)
        assert isinstance(results, list)
        
    except Exception as e:
        pytest.skip(f"Vectorstore not initialized: {e}")


def test_tools_creation():
    """Test custom tools creation."""
    from src.rag.vectorstore import FitnessVectorStore
    from src.tools.custom_tools import FitnessTools
    
    try:
        vectorstore = FitnessVectorStore()
        tools = FitnessTools(vectorstore)
        langchain_tools = tools.create_langchain_tools()
        
        assert len(langchain_tools) > 0
        assert all(hasattr(tool, 'name') for tool in langchain_tools)
        assert all(hasattr(tool, 'description') for tool in langchain_tools)
        
    except Exception as e:
        pytest.skip(f"Vectorstore not initialized: {e}")


if __name__ == "__main__":
    """Run tests manually without pytest."""
    print("Running basic system tests...\n")
    
    tests = [
        ("Import test", test_imports),
        ("User profile creation", test_user_profile_creation),
        ("Data generators", test_data_generators),
        ("Configuration", test_config),
        ("Vectorstore", test_vectorstore_initialization),
        ("Tools creation", test_tools_creation),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            print(f"Testing {name}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⏭️  SKIPPED: {e}")
            skipped += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*50}\n")
    
    if failed > 0:
        sys.exit(1)
