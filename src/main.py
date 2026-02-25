import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .app import rag_with_history # Импортируем твою логику из app.py

app = FastAPI(title="Legal AI API (Uzbekistan)")

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default_user"

@app.post("/ask")
async def ask_legal_bot(request: QuestionRequest):
    try:
        config = {"configurable": {"session_id": request.session_id}}
        # Вызываем твою цепочку RAG
        response = rag_with_history.invoke({"question": request.question}, config=config)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)