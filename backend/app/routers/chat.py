from typing import Any, Dict
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.deps import require_service_api_key

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(require_service_api_key)],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

from fastapi.responses import StreamingResponse

@router.post("/", response_model=Dict[str, Any])
async def post_chat(request: Request, payload: ChatRequest) -> Dict[str, Any]:
    """Process a chat message in the context of a session."""
    manager = request.app.state.chat_manager
    return await manager.run_chat(session_id=payload.session_id, message=payload.message)

@router.post("/stream")
async def stream_chat(
    request: Request,
    payload: ChatRequest,
):
    """Stream a chat response using SSE."""
    manager = request.app.state.chat_manager

    async def event_generator():
        async for chunk in manager.stream_chat(payload.session_id, payload.message):
            if await request.is_disconnected():
                break
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
