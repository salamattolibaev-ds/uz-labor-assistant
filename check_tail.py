from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

data = db.get()
num = len(data['documents'])
print(f"Всего фрагментов: {num}")

# Выводим последние 2 фрагмента целиком
print("\n--- ПОСЛЕДНИЙ ФРАГМЕНТ (№" + str(num) + ") ---")
print(data['documents'][-1])
print("\n--- ПРЕДПОСЛЕДНИЙ ФРАГМЕНТ (№" + str(num-1) + ") ---")
print(data['documents'][-2])