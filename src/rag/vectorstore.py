"""Vector store management using ChromaDB."""
import json
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


class FitnessVectorStore:
    """Manages vector stores for fitness knowledge base."""
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """Initialize ChromaDB client and embedding model."""
        if persist_directory is None:
            persist_directory = config.CHROMA_DIR
        
        self.client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
        
        # Initialize collections
        self.collections = {}
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Create or get existing collections."""
        for key, name in config.COLLECTIONS.items():
            try:
                self.collections[key] = self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"✅ Collection '{name}' ready")
            except Exception as e:
                print(f"❌ Error with collection '{name}': {e}")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def index_exercises(self, exercises: List[Dict]):
        """Index exercises into vector store."""
        collection = self.collections["exercises"]
        
        # Clear existing data
        try:
            collection_ids = collection.get()["ids"]
            if collection_ids:
                collection.delete(ids=collection_ids)
        except:
            pass
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, exercise in enumerate(tqdm(exercises, desc="Indexing exercises")):
            # Create rich text representation for embedding
            doc_text = f"""
            Упражнение: {exercise['name']}
            Мышечные группы: {', '.join(exercise['muscle_groups'])}
            Оборудование: {', '.join(exercise['equipment'])}
            Сложность: {exercise['difficulty']}
            Описание: {exercise['description']}
            Техника: {' '.join(exercise['technique_points'])}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                "name": exercise["name"],
                "muscle_groups": json.dumps(exercise["muscle_groups"], ensure_ascii=False),
                "equipment": json.dumps(exercise["equipment"], ensure_ascii=False),
                "difficulty": exercise["difficulty"],
                "description": exercise["description"],
                "technique_points": json.dumps(exercise["technique_points"], ensure_ascii=False),
                "video_keywords": json.dumps(exercise.get("video_keywords", []), ensure_ascii=False)
            })
            ids.append(f"exercise_{idx}")
        
        # Generate embeddings
        embeddings = self._embed_texts(documents)
        
        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(exercises)} exercises")
    
    def index_supplements(self, supplements: List[Dict]):
        """Index supplements into vector store."""
        collection = self.collections["supplements"]
        
        # Clear existing data
        try:
            collection_ids = collection.get()["ids"]
            if collection_ids:
                collection.delete(ids=collection_ids)
        except:
            pass
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, supplement in enumerate(tqdm(supplements, desc="Indexing supplements")):
            doc_text = f"""
            Добавка: {supplement['name']}
            Категория: {supplement['category']}
            Преимущества: {' '.join(supplement['benefits'])}
            Дозировка: {supplement['dosage']}
            Время приема: {supplement['timing']}
            Уровень доказательств: {supplement['evidence_level']}
            Заметки: {supplement.get('notes', '')}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                "name": supplement["name"],
                "category": supplement["category"],
                "benefits": json.dumps(supplement["benefits"], ensure_ascii=False),
                "dosage": supplement["dosage"],
                "timing": supplement["timing"],
                "contraindications": json.dumps(supplement["contraindications"], ensure_ascii=False),
                "evidence_level": supplement["evidence_level"],
                "notes": supplement.get("notes", ""),
                "interactions": supplement.get("interactions", "")
            })
            ids.append(f"supplement_{idx}")
        
        embeddings = self._embed_texts(documents)
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(supplements)} supplements")
    
    def index_nutrition(self, nutrition_articles: List[Dict]):
        """Index nutrition knowledge into vector store."""
        collection = self.collections["nutrition"]
        
        # Clear existing data
        try:
            collection_ids = collection.get()["ids"]
            if collection_ids:
                collection.delete(ids=collection_ids)
        except:
            pass
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, article in enumerate(tqdm(nutrition_articles, desc="Indexing nutrition")):
            doc_text = f"""
            Тема: {article['topic']}
            {article['content']}
            """
            
            documents.append(doc_text.strip())
            metadatas.append({
                "topic": article["topic"],
                "content": article["content"],
                "tags": json.dumps(article["tags"], ensure_ascii=False)
            })
            ids.append(f"nutrition_{idx}")
        
        embeddings = self._embed_texts(documents)
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(nutrition_articles)} nutrition articles")
    
    def search_exercises(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for exercises."""
        collection = self.collections["exercises"]
        
        query_embedding = self._embed_texts([query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters
        )
        
        return self._format_results(results)
    
    def search_supplements(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """Search for supplements."""
        collection = self.collections["supplements"]
        
        query_embedding = self._embed_texts([query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return self._format_results(results)
    
    def search_nutrition(
        self,
        query: str,
        n_results: int = 3
    ) -> List[Dict]:
        """Search nutrition knowledge."""
        collection = self.collections["nutrition"]
        
        query_embedding = self._embed_texts([query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into list of dicts."""
        formatted = []
        
        if not results["ids"] or not results["ids"][0]:
            return formatted
        
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i]
            }
            formatted.append(result)
        
        return formatted


def initialize_vectorstore_with_data():
    """Initialize vector store and load all data."""
    from src.data.generators import (
        generate_exercises_data,
        generate_supplements_data,
        generate_nutrition_data
    )
    
    print("\n🚀 Initializing Fitness AI Vector Store")
    print("=" * 50)
    
    # Create vector store
    vectorstore = FitnessVectorStore()
    
    # Generate and index data
    print("\n📊 Generating data...")
    exercises = generate_exercises_data()
    supplements = generate_supplements_data()
    nutrition = generate_nutrition_data()
    
    print("\n🔍 Indexing into ChromaDB...")
    vectorstore.index_exercises(exercises)
    vectorstore.index_supplements(supplements)
    vectorstore.index_nutrition(nutrition)
    
    print("\n✅ Vector store initialization complete!")
    print("=" * 50)
    
    return vectorstore


if __name__ == "__main__":
    vectorstore = initialize_vectorstore_with_data()