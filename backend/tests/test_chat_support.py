from manager.chat_support import (
    build_chat_prompt,
    build_unavailable_payload,
    detect_response_language,
    get_chat_error_message,
    get_stream_error_message,
)


def test_detect_response_language_english():
    assert detect_response_language("I have chest pain") == "en"


def test_detect_response_language_arabic():
    assert detect_response_language("عندي ألم في الصدر") == "ar"


def test_build_unavailable_payload_english_message():
    payload = build_unavailable_payload("s1", "Can you explain this result?")
    assert "GEMINI_API_KEY" in payload["response"]
    assert "unavailable" in payload["response"].lower()


def test_build_unavailable_payload_arabic_message():
    payload = build_unavailable_payload("s2", "ممكن تشرح النتيجة؟")
    assert "غير متاحة" in payload["response"]


def test_chat_error_message_is_language_aware():
    assert "error" in get_chat_error_message("Can you help me?").lower()
    assert "حدث خطأ" in get_chat_error_message("محتاج مساعدة")


def test_stream_error_message_is_language_aware():
    assert "stream" in get_stream_error_message("stream please").lower()
    assert "بث" in get_stream_error_message("رد بالبث")


def test_build_chat_prompt_mentions_language_policy():
    prompt = build_chat_prompt(
        [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi"},
        ]
    )
    assert "same language" in prompt.lower()
    assert "User: Hello" in prompt
