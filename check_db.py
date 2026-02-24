from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Подключаем те же настройки, что и в основном боте
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Получаем данные из базы
data = db.get()
num_chunks = len(data['documents'])

print(f"--- АНАЛИЗ БАЗЫ ДАННЫХ ---")
print(f"Всего фрагментов найдено: {num_chunks}")

if num_chunks > 0:
    print("\n--- СОДЕРЖАНИЕ ПОСЛЕДНИХ 3 ФРАГМЕНТОВ ---")
    # Берем последние 3 кусочка
    for i in range(1, 4):
        print(f"\nФрагмент №{num_chunks - i + 1}:")
        print(data['documents'][-i][:500] + "...") # Печатаем первые 500 символов каждого
else:
    print("База данных пуста!")