from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import (
    ChatHistoryItem,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionCreateRequest,
    ChatSessionSummary,
)
from app.state import AppServices

router = APIRouter(prefix="/chat", tags=["chat"])


def get_services(request: Request) -> AppServices:
    return request.app.state.services


@router.post("/message", response_model=ChatMessageResponse)
def send_message(payload: ChatMessageRequest, services: AppServices = Depends(get_services)) -> ChatMessageResponse:
    session_id = services.chat_repo.ensure_session(payload.session_id, payload.user_id, payload.user_name)

    user_message_id = services.chat_repo.add_message(session_id, "user", payload.message)
    try:
        triplets = services.llm_extractor.extract_triplets(payload.message, payload.user_name)
        stored_triplets = services.graph_repo.upsert_triplets(
            triplets=triplets,
            source_message_id=str(user_message_id),
            source_message=payload.message,
        )
        assistant_message = services.llm_extractor.build_assistant_reply(payload.message, payload.user_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    services.chat_repo.add_message(session_id, "assistant", assistant_message)

    return ChatMessageResponse(
        session_id=session_id,
        user_name=payload.user_name,
        assistant_message=assistant_message,
        extracted_triplets=triplets,
        stored_triplets_count=stored_triplets,
    )


@router.post("/sessions", response_model=ChatSessionSummary)
def create_session(payload: ChatSessionCreateRequest, services: AppServices = Depends(get_services)) -> ChatSessionSummary:
    session = services.chat_repo.create_session(payload.user_name)
    return ChatSessionSummary(**session)


@router.get("/sessions", response_model=list[ChatSessionSummary])
def list_sessions(limit: int = 100, services: AppServices = Depends(get_services)) -> list[ChatSessionSummary]:
    sessions = services.chat_repo.list_sessions(limit=limit)
    return [ChatSessionSummary(**item) for item in sessions]


@router.get("/history", response_model=list[ChatHistoryItem])
def get_history(session_id: str, limit: int = 100, services: AppServices = Depends(get_services)) -> list[ChatHistoryItem]:
    history = services.chat_repo.get_history(session_id=session_id, limit=limit)
    return [ChatHistoryItem(**item) for item in history]
