import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import (
    ChatHistoryItem,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionCreateRequest,
    ChatSessionSummary,
)
from app.state import AppServices
from loguru import logger

router = APIRouter(prefix="/chat", tags=["chat"])


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def format_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/message", response_model=ChatMessageResponse)
def send_message(payload: ChatMessageRequest, services: AppServices = Depends(get_services)) -> ChatMessageResponse:
    request_id = uuid4().hex[:8]
    session_id = services.chat_repo.ensure_session(payload.session_id, payload.user_id, payload.user_name)

    logger.info(
        "chat_message_start request_id={} session_id={} user_name={} message_len={} profile={}",
        request_id,
        session_id,
        payload.user_name,
        len(payload.message),
        payload.prompt_profile,
    )

    history = services.chat_repo.get_history(session_id, limit=30)
    user_message_id = services.chat_repo.add_message(session_id, "user", payload.message)
    try:
        triplets = services.llm_extractor.extract_and_stage(
            message=payload.message,
            user_name=payload.user_name,
            prompt_profile=payload.prompt_profile,
            graph_repo=services.graph_repo,
            history=history,
        )
        stored_triplets = services.graph_repo.upsert_triplets(
            triplets=triplets,
            source_message_id=str(user_message_id),
            source_message=payload.message,
        )
        assistant_message = services.llm_extractor.build_assistant_reply(
            payload.message,
            payload.user_name,
            payload.prompt_profile,
            history,
        )
    except Exception as exc:
        logger.exception(
            "chat_message_error request_id={} session_id={} exception_type={}",
            request_id,
            session_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    services.chat_repo.add_message(session_id, "assistant", assistant_message)

    return ChatMessageResponse(
        session_id=session_id,
        user_name=payload.user_name,
        prompt_profile=payload.prompt_profile,
        assistant_message=assistant_message,
        extracted_triplets=triplets,
        stored_triplets_count=stored_triplets,
    )


@router.post("/message/stream")
def send_message_stream(payload: ChatMessageRequest, services: AppServices = Depends(get_services)) -> StreamingResponse:
    request_id = uuid4().hex[:8]
    session_id = services.chat_repo.ensure_session(payload.session_id, payload.user_id, payload.user_name)

    logger.info(
        "chat_stream_start request_id={} session_id={} user_name={} message_len={} profile={}",
        request_id,
        session_id,
        payload.user_name,
        len(payload.message),
        payload.prompt_profile,
    )

    history = services.chat_repo.get_history(session_id, limit=30)
    user_message_id = services.chat_repo.add_message(session_id, "user", payload.message)
    try:
        triplets = services.llm_extractor.extract_and_stage(
            message=payload.message,
            user_name=payload.user_name,
            prompt_profile=payload.prompt_profile,
            graph_repo=services.graph_repo,
            history=history,
        )
        stored_triplets = services.graph_repo.upsert_triplets(
            triplets=triplets,
            source_message_id=str(user_message_id),
            source_message=payload.message,
        )
    except Exception as exc:
        logger.exception(
            "chat_stream_extraction_error request_id={} session_id={} exception_type={}",
            request_id,
            session_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    def event_stream():
        collected_tokens: list[str] = []
        try:
            for token in services.llm_extractor.stream_assistant_reply(
                payload.message,
                payload.user_name,
                payload.prompt_profile,
                history,
            ):
                collected_tokens.append(token)
                yield format_sse("token", {"text": token})

            assistant_message = "".join(collected_tokens).strip() or "Posso ajudar a detalhar isso melhor se quiser."
            services.chat_repo.add_message(session_id, "assistant", assistant_message)

            response_payload = ChatMessageResponse(
                session_id=session_id,
                user_name=payload.user_name,
                prompt_profile=payload.prompt_profile,
                assistant_message=assistant_message,
                extracted_triplets=triplets,
                stored_triplets_count=stored_triplets,
            ).model_dump()
            logger.info(
                "chat_stream_done request_id={} session_id={} triplets={} tokens={}",
                request_id,
                session_id,
                len(triplets),
                len(collected_tokens),
            )
            yield format_sse("done", {**response_payload, "request_id": request_id})
        except Exception as exc:
            logger.exception(
                "chat_stream_runtime_error request_id={} session_id={} exception_type={} tokens_collected={}",
                request_id,
                session_id,
                type(exc).__name__,
                len(collected_tokens),
            )
            yield format_sse("error", {
                "detail": f"LLM stream failed: {exc}",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
