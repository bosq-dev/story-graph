from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import ChatHistoryItem, ChatMessageRequest, ChatMessageResponse
from app.state import AppServices

router = APIRouter(prefix="/chat", tags=["chat"])


def get_services(request: Request) -> AppServices:
    return request.app.state.services


@router.post("/message", response_model=ChatMessageResponse)
def send_message(payload: ChatMessageRequest, services: AppServices = Depends(get_services)) -> ChatMessageResponse:
    session_id = services.chat_repo.ensure_session(payload.session_id, payload.user_id)

    user_message_id = services.chat_repo.add_message(session_id, "user", payload.message)
    try:
        triplets = services.llm_extractor.extract_triplets(payload.message)
        stored_triplets = services.graph_repo.upsert_triplets(
            triplets=triplets,
            source_message_id=str(user_message_id),
            source_message=payload.message,
        )
        assistant_message = services.llm_extractor.build_assistant_reply(payload.message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    services.chat_repo.add_message(session_id, "assistant", assistant_message)

    return ChatMessageResponse(
        session_id=session_id,
        assistant_message=assistant_message,
        extracted_triplets=triplets,
        stored_triplets_count=stored_triplets,
    )


@router.get("/history", response_model=list[ChatHistoryItem])
def get_history(session_id: str, limit: int = 100, services: AppServices = Depends(get_services)) -> list[ChatHistoryItem]:
    history = services.chat_repo.get_history(session_id=session_id, limit=limit)
    return [ChatHistoryItem(**item) for item in history]
