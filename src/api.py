from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import (
    load_vectorstore,
    load_reranker,
    answer_question
)

app = FastAPI(
    title="Financial RAG API",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    ticker: str
    form_type: str
    filing_date: str
    section: str
    chunk_id: int


class QueryResponse(BaseModel):
    question: str
    ticker: str | None
    selected_section: str | None
    retrieval_rounds: int
    additional_section: str | None
    answer: str
    sources: list[SourceResponse]


vectorstore = load_vectorstore()
reranker = load_reranker()


@app.get("/")
def root():
    return {
        "message": "Financial RAG API is running"
    }


@app.post(
    "/query",
    response_model=QueryResponse
)
def query(request: QueryRequest):
    result = answer_question(
        request.question,
        vectorstore,
        reranker
    )

    return result