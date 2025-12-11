import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

# Параметры для тренера
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_FILE = os.path.join(CURRENT_DIR, "../data/knowledge/TrainerData.pdf")
TRAINER_DB_PATH = os.path.join(CURRENT_DIR, "../db/trainer_vectordb")

# Функция для чтения PDF
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Функция создания базы для тренера
def build_trainer_vectordb(file_path, db_path, embeddings_model):
    if os.path.exists(db_path):
        print(f"Загружаем существующую базу тренера: {db_path}")
        return FAISS.load_local(db_path, embeddings_model)

    # Читаем PDF
    text = read_pdf(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

    # Делим на куски
    # splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Создаём эмбеддинги
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Создаём FAISS базу
    vectordb = FAISS.from_texts(chunks, embeddings)

    # Сохраняем на диск
    os.makedirs(db_path, exist_ok=True)
    vectordb.save_local(db_path)
    print(f"База тренера сохранена: {db_path}")
    return vectordb

# Основная функция
def setup_trainer_database():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    trainer_db = build_trainer_vectordb(TRAINER_FILE, TRAINER_DB_PATH, embeddings_model)
    return trainer_db

if __name__ == "__main__":
    setup_trainer_database()
