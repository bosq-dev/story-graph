import json
import logging
from datetime import datetime, timezone
from uuid import uuid4
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.schemas import AdminChatRequest
from app.services.admin_graph_tools import AdminGraphTools
from app.state import AppServices

router = APIRouter(prefix="/admin/chat", tags=["admin-chat"])
logger = logging.getLogger(__name__)


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def format_sse(event: str, payload: dict) -> str:
    # `default=str` prevents stream crashes with non-JSON-native values returned by Neo4j.
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def chunk_text(text: str, size: int = 60) -> Iterator[str]:
    if not text:
        return
    for index in range(0, len(text), size):
        yield text[index : index + size]


@router.post("/message/stream")
def send_admin_message_stream(
    payload: AdminChatRequest,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    request_id = uuid4().hex[:8]
    session_id = services.chat_repo.ensure_session(payload.session_id, payload.user_id, payload.user_name)
    history = services.chat_repo.get_history(session_id, limit=30)
    services.chat_repo.add_message(session_id, "user", payload.message)

    graph_tools = AdminGraphTools(graph_repo=services.graph_repo)
    logger.info(
        "admin_stream_start request_id=%s session_id=%s user_name=%s history_items=%s",
        request_id,
        session_id,
        payload.user_name,
        len(history),
    )

    def event_stream():
        try:
            outcome = services.llm_extractor.run_admin_assistant_with_tools(
                message=payload.message,
                user_name=payload.user_name,
                history=history,
                graph_tools=graph_tools,
            )
            tool_calls = outcome.get("tool_calls", [])
            tool_results = outcome.get("tool_results", [])

            for item in tool_calls:
                yield format_sse("tool_call", item)

            for item in tool_results:
                yield format_sse("tool_result", item)

            assistant_message = str(outcome.get("assistant_message", "")).strip()
            if not assistant_message:
                assistant_message = "Nao consegui concluir com dados suficientes no grafo."

            for token in chunk_text(assistant_message):
                yield format_sse("token", {"text": token})

            services.chat_repo.add_message(session_id, "assistant", assistant_message)
            logger.info(
                "admin_stream_done request_id=%s session_id=%s tool_calls=%s tool_results=%s assistant_len=%s",
                request_id,
                session_id,
                len(tool_calls),
                len(tool_results),
                len(assistant_message),
            )
            yield format_sse(
                "done",
                {
                    "session_id": session_id,
                    "user_name": payload.user_name,
                    "assistant_message": assistant_message,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "request_id": request_id,
                },
            )
        except Exception as exc:
            logger.exception("admin_stream_error request_id=%s session_id=%s", request_id, session_id)
            yield format_sse(
                "error",
                {
                    "detail": f"Admin stream failed: {exc}",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
