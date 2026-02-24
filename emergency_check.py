from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Проверяем именно ту модель, которая должна быть везде
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

data = db.get()
print(f"Фрагментов в базе: {len(data['documents'])}")

if len(data['documents']) > 0:
    print("\n--- ПЕРВЫЙ ФРАГМЕНТ В БАЗЕ ---")
    print(data['documents'][0][:500]) # Посмотрим начало кодекса
else:
    print("БАЗА ПУСТА!")