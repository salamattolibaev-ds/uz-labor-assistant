import os
from operator import itemgetter
from dotenv import load_dotenv

# Используем Groq для скорости и точности
from langchain_groq import ChatGroq 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# 1. Инициализация базы данных и эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# k=15 позволяет модели видеть больше контекста для качественного сравнения
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 15})

# 2. Инициализация модели Groq (Llama 3 70B — идеальна для логических тестов)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# 3. Юридически защищенный промпт (Твоя обновленная версия)
system_prompt = """Ты — экспертный юридический справочный ассистент по Трудовому кодексу Республики Узбекистан.

⚠️ ДИСКЛЕЙМЕР: Ты не являешься адвокатом. Твои ответы носят исключительно информационный характер. Для принятия юридически значимых решений пользователю необходимо обратиться к квалифицированному юристу.

ПРАВИЛА АНОНИМНОСТЬЮ:
Если пользователь вводит личные данные (ФИО, ИНН, названия компаний), вежливо напомни в начале ответа, что в целях безопасности лучше использовать вымышленные имена.

Инструкция для анализа:
1. ФАКТЫ: Четко выпиши из вопроса даты, сроки, действия и участников.
2. НОРМЫ: Процитируй или перескажи подходящие статьи из предоставленного Контекста.
3. СРАВНЕНИЕ (ЛОГИЧЕСКИЙ ТЕСТ): Проведи прямое математическое или смысловое сравнение факта и нормы.
4. ВЕРДИКТ: Сформулируй краткий и обоснованный итог.

ПРИМЕР ТВОЕЙ ЛОГИКИ:
Вопрос: Прошло 5 дней с момента увольнения. Закон (ст. X) дает 3 дня на иск.
Анализ: 5 дней (факт) > 3 дня (норма закона). 
Результат: Срок обращения пропущен.

Контекст для поиска:
{context}

Теперь проанализируй вопрос пользователя ниже. 
В конце обязательно укажи: [Источник: База данных ТК РУз]."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 4. Вспомогательные функции для RAG
def format_docs(docs):
    found_articles = [str(doc.metadata.get('article_number', 'б/н')) for doc in docs]
    print(f"\n📍 Извлечены статьи для анализа: {', '.join(set(found_articles))}")
    return "\n\n".join(doc.page_content for doc in docs)

# Цепочка обработки (LCEL)
core_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Управление историей чата
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

# 6. Функция запуска с прозрачностью интерфейса
def start_bot():
    print("\n" + "="*50)
    print("🏛️  LEGAL AI SYSTEM (LCEL + 70B) — УЗБЕКИСТАН")
    print("="*50)
    print("🛡️  ПРАВОВАЯ ИНФОРМАЦИЯ И БЕЗОПАСНОСТЬ:")
    print("1. Бот работает в режиме справочной системы.")
    print("2. Ответы базируются на официальном Трудовом кодексе РУз.")
    print("3. ПОЖАЛУЙСТА, НЕ ВВОДИТЕ РЕАЛЬНЫЕ ФИО И АДРЕСА.")
    print("4. ИИ может ошибаться — всегда проверяйте важные выводы у юристов.")
    print("="*50)

    session_id = "default_user"
    
    while True:
        user_input = input("\n⚖️ Ваш вопрос (или 'exit'): ")
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break

        if user_input.strip():
            print("🔍 Идет юридический анализ...")
            try:
                config = {"configurable": {"session_id": session_id}}
                print("\nАНАЛИЗ И ВЕРДИКТ:\n")
                
                # Потоковый вывод ответа (Streaming)
                for chunk in rag_with_history.stream({"question": user_input}, config=config):
                    print(chunk, end="", flush=True)
                print("\n" + "-"*30)
                
            except Exception as e:
                print(f"\n❌ Ошибка системы: {e}")

if __name__ == "__main__":
    start_bot()