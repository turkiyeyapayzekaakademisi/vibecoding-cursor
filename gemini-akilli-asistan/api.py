"""Tarih ve kültür asistanı için FastAPI arayüzü."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from ollama import Client, ResponseError
from pydantic import BaseModel, Field, ValidationError

from assistant import (
    TarihKulturYaniti,
    ensure_ready,
    generate_answer,
    load_settings,
    new_history,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="Kullanıcı sorusu")
    conversation_id: str | None = Field(
        default=None,
        description="Devam eden sohbet kimliği. Boşsa yeni sohbet açılır.",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    answer: TarihKulturYaniti


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama: str


class ConversationDeleted(BaseModel):
    conversation_id: str
    deleted: bool


host, model = load_settings()
client = Client(host=host)
conversations: dict[str, list[dict[str, str]]] = {}

app = FastAPI(
    title="Tarih ve Kültür Asistanı",
    description="Ollama qwen3:4b ile structured JSON üreten sohbet API'si.",
    version="1.0.0",
)


def get_or_create_history(conversation_id: str | None) -> tuple[str, list[dict[str, str]]]:
    if not conversation_id:
        new_id = str(uuid4())
        conversations[new_id] = new_history()
        return new_id, conversations[new_id]

    history = conversations.get(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Sohbet bulunamadı.")
    return conversation_id, history


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = ensure_ready(client, model)
    return HealthResponse(
        status="ok" if ready else "ollama_unavailable",
        model=model,
        ollama=host,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not ensure_ready(client, model):
        raise HTTPException(status_code=503, detail="Ollama hazır değil veya model yüklü değil.")

    conversation_id, history = get_or_create_history(payload.conversation_id)
    try:
        answer = generate_answer(client, model, history, payload.message.strip())
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail=f"JSON şeması bozuldu: {exc}") from exc
    except ResponseError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama hatası: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(conversation_id=conversation_id, answer=answer)


@app.delete("/conversations/{conversation_id}", response_model=ConversationDeleted)
def delete_conversation(conversation_id: str) -> ConversationDeleted:
    deleted = conversations.pop(conversation_id, None) is not None
    if not deleted:
        raise HTTPException(status_code=404, detail="Sohbet bulunamadı.")
    return ConversationDeleted(conversation_id=conversation_id, deleted=True)
