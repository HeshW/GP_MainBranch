from __future__ import annotations

import re
from typing import Dict, Iterable, List

ChatMessage = Dict[str, str]
ChatHistory = List[ChatMessage]

SYSTEM_INSTRUCTION = (
    "You are a medical assistant for this application. "
    "Always respond in the same language as the user's latest message: "
    "English if the user writes in English, Arabic if the user writes in Arabic. "
    "Keep responses concise, clear, and clinically safe. "
    "Do not provide a definitive diagnosis or direct medication prescriptions. "
    "Use only the information available in the conversation and recommend clinician review, "
    "especially for severe, urgent, or worsening symptoms. "
    "If asked who built you, answer exactly: 'I was developed by Mr.Bondo2'."
)

UNAVAILABLE_MESSAGE_EN = (
    "Chat service is currently unavailable because GEMINI_API_KEY is missing or invalid. "
    "Please configure the key and try again."
)

UNAVAILABLE_MESSAGE_AR = (
    "خدمة الدردشة غير متاحة حاليا لأن GEMINI_API_KEY غير مضبوط أو غير صالح. "
    "يرجى ضبط المفتاح ثم إعادة المحاولة."
)

CHAT_ERROR_MESSAGE_EN = (
    "An error occurred while generating the reply. Please try again shortly."
)

CHAT_ERROR_MESSAGE_AR = (
    "حدث خطأ أثناء توليد الرد. يرجى المحاولة مرة أخرى بعد قليل."
)

STREAM_ERROR_MESSAGE_EN = "A streaming error occurred. Please retry."

STREAM_ERROR_MESSAGE_AR = "حدث خطأ أثناء بث الرد. يرجى إعادة المحاولة."


def _contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", str(text or "")))


def detect_response_language(message: str) -> str:
    return "ar" if _contains_arabic(message) else "en"


def get_unavailable_message(message: str) -> str:
    return UNAVAILABLE_MESSAGE_AR if detect_response_language(message) == "ar" else UNAVAILABLE_MESSAGE_EN


def get_chat_error_message(message: str) -> str:
    return CHAT_ERROR_MESSAGE_AR if detect_response_language(message) == "ar" else CHAT_ERROR_MESSAGE_EN


def get_stream_error_message(message: str) -> str:
    return STREAM_ERROR_MESSAGE_AR if detect_response_language(message) == "ar" else STREAM_ERROR_MESSAGE_EN


def build_chat_prompt(history: Iterable[ChatMessage], *, limit: int = 8) -> str:
    recent_messages = list(history)[-limit:]
    context_lines = []
    for message in recent_messages:
        role_tag = "User" if message["role"] == "user" else "Assistant"
        context_lines.append(f"{role_tag}: {message['content']}")
    context = "\n".join(context_lines)
    return (
        "Conversation context:\n"
        f"{context}\n"
        "Write the next medical assistant reply in the same language as the user's latest message (English or Arabic):"
    )


def build_unavailable_payload(session_id: str, message: str) -> Dict[str, str]:
    return {
        "session_id": session_id,
        "message": message,
        "response": get_unavailable_message(message),
    }
