from fastapi import FastAPI
from fastapi import HTTPException

from rag.rag_engine import RAGEngine
from api.schemas import (
    QuestionRequest,
    AnswerResponse
)

engine = RAGEngine()

app = FastAPI(
    title="API pour les évènements culturels",
    description="API de recommandation d'événements culturels",
    version="1.0"
)


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask_question(
    request: QuestionRequest
):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question vide"
        )

    answer = engine.ask(
        request.question
    )

    return {
        "answer": answer
    }


@app.post("/rebuild")
def rebuild():

    result = engine.rebuild()

    return result