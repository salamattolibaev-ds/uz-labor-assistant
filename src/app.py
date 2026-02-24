import os
from operator import itemgetter
from dotenv import load_dotenv

# Меняем Groq на Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# 1. Инициализация базы
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# Для Gemini можно оставить k=15, она легко справляется с таким объемом
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 15})

# 2. Инициализация Gemini 3 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=True
)

# 3. Юридический промпт (Твой вариант с примером 5 > 3 — лучший!)
system_prompt = """Ты — экспертный юридический справочный ассистент по Трудовому кодексу Республики Узбекистан.

⚠️ ДИСКЛЕЙМЕР: Ты не являешься адвокатом. Твои ответы носят справочный характер. 
🛡️ АНОНИМНОСТЬ: Если в вопросе есть ФИО или названия компаний, напомни пользователю использовать вымышленные данные.

Инструкция для анализа:
1. ФАКТЫ: Четко выпиши из вопроса даты, сроки, действия и участников.
2. НОРМЫ: Процитируй или перескажи статьи из Контекста.
3. СРАВНЕНИЕ (ЛОГИЧЕСКИЙ ТЕСТ): Сравни факты и нормы (Пример: 5 дней > 3 дня).
4. ВЕРДИКТ: Сформулируй краткий итог.

Контекст:
{context}

[Источник: База данных ТК РУз]"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def format_docs(docs):
    found_articles = [str(doc.metadata.get('article_number', 'б/н')) for doc in docs]
    print(f"\n📍 Ретривер извлек статьи: {', '.join(set(found_articles))}")
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Цепочка (Chain)
core_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt
    | llm
    | StrOutputParser()
)

history_store = {}

def get_session_history(session_id: str):
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

rag_with_history = RunnableWithMessageHistory(
    core_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 5. Функция запуска
def start_bot():
    print("\n" + "="*50)
    print("🏛️  LEGAL AI SYSTEM (Gemini 3 Flash) — УЗБЕКИСТАН")
    print("="*50)
    print("🛡️  ПРАВОВАЯ ИНФОРМАЦИЯ И БЕЗОПАСНОСТЬ:")
    print("1. Режим справочной системы.")
    print("2. Данные на базе ТК РУз.")
    print("3. БЕЗ РЕАЛЬНЫХ ПЕРСОНАЛЬНЫХ ДАННЫХ.")
    print("="*50)

    session_id = "default_user"
    
    while True:
        user_input = input("\n⚖️ Ваш вопрос (или 'exit'): ")
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break

        if user_input.strip():
            print("🔍 Поиск и анализ...")
            try:
                config = {"configurable": {"session_id": session_id}}
                print("\nОтвет: ", end="", flush=True)
                
                # Streaming работает через Gemini так же отлично
                for chunk in rag_with_history.stream({"question": user_input}, config=config):
                    print(chunk, end="", flush=True)
                print("\n" + "-"*30)
                
            except Exception as e:
                print(f"\n❌ Ошибка системы: {e}")

if __name__ == "__main__":
    start_bot()