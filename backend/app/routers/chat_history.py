from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import ChatMessage, ChatSession, User
from app.schemas.chat_history import ChatMessageCreate, ChatMessageRead, ChatSessionCreate, ChatSessionRead
from app.security import get_current_user

router = APIRouter(prefix="/chats", tags=["Chat History"])


def _get_owned_chat(chat_id: int, user_id: int, db: Session) -> ChatSession:
    chat = db.query(ChatSession).filter(ChatSession.id == chat_id, ChatSession.user_id == user_id).first()
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")
    return chat


def _default_title(content: str) -> str:
    title = " ".join(content.strip().split())
    if len(title) > 60:
        return f"{title[:57].rstrip()}..."
    return title or "New chat"


@router.get("", response_model=list[ChatSessionRead])
def list_chats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatSession]:
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc())
        .all()
    )


@router.post("", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
def create_chat(
    payload: ChatSessionCreate | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatSession:
    raw_title = payload.title if payload else None
    chat = ChatSession(user_id=current_user.id, title=(raw_title or "New chat").strip() or "New chat")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    chat = _get_owned_chat(chat_id, current_user.id, db)
    db.delete(chat)
    db.commit()


@router.get("/{chat_id}/messages", response_model=list[ChatMessageRead])
def list_messages(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatMessage]:
    _get_owned_chat(chat_id, current_user.id, db)
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.chat_session_id == chat_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )


@router.post("/{chat_id}/messages", response_model=ChatMessageRead, status_code=status.HTTP_201_CREATED)
def create_message(
    chat_id: int,
    payload: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatMessage:
    chat = _get_owned_chat(chat_id, current_user.id, db)
    message = ChatMessage(chat_session_id=chat.id, role=payload.role, content=payload.content)
    if payload.role == "user" and chat.title == "New chat":
        chat.title = _default_title(payload.content)
    chat.updated_at = datetime.now(timezone.utc)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message
