import { useState } from "react";
import { postChat, postChatStream } from "@/shared/api";
import { ChatMessage } from "@/shared/types";

function createSessionId() {
  return `session-${Math.random().toString(36).slice(2, 10)}`;
}

export function ChatInterface() {
  const [sessionId] = useState(createSessionId);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const appendChunkToLastModel = (chunk: string) => {
    setMessages((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "model") {
          next[index] = {
            ...next[index],
            content: `${next[index].content}${chunk}`,
          };
          return next;
        }
      }
      return [...next, { role: "model", content: chunk }];
    });
  };

  const setLastModelMessage = (content: string) => {
    setMessages((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "model") {
          next[index] = { ...next[index], content };
          return next;
        }
      }
      return [...next, { role: "model", content }];
    });
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    setInput("");
    setMessages((current) => [
      ...current,
      { role: "user", content: trimmed },
      { role: "model", content: "" },
    ]);
    setIsLoading(true);

    let hasStreamedChunk = false;

    try {
      const streamedText = await postChatStream(
        { session_id: sessionId, message: trimmed },
        (chunk) => {
          hasStreamedChunk = true;
          appendChunkToLastModel(chunk);
        },
      );

      if (!hasStreamedChunk && !streamedText) {
        const fallback = await postChat({ session_id: sessionId, message: trimmed });
        setLastModelMessage(fallback.response);
      }
    } catch {
      if (!hasStreamedChunk) {
        try {
          const fallback = await postChat({ session_id: sessionId, message: trimmed });
          setLastModelMessage(fallback.response);
        } catch (error) {
          const content = error instanceof Error ? error.message : String(error);
          setLastModelMessage(`Error: ${content}`);
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-card" dir="rtl">
      <h3 className="chat-card__title">الدردشة الطبية</h3>
      <p className="chat-card__subtitle">اسأل عن النتيجة الحالية أو اطلب توضيحًا إضافيًا بشكل مبسط.</p>

      <div className="chat-history">
        {messages.length === 0 ? (
          <p className="chat-empty">لا توجد رسائل بعد. يمكنك كتابة سؤالك عن التحليل أو التشخيص.</p>
        ) : (
          messages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`chat-bubble ${message.role === "user" ? "chat-bubble--user" : "chat-bubble--model"}`}
            >
              <strong className="chat-bubble__role">
                {message.role === "user" ? "أنت" : "المساعد الطبي"}
              </strong>
              <span className="chat-bubble__content">{message.content}</span>
            </article>
          ))
        )}

        {isLoading && <div className="chat-status">جارٍ توليد الرد...</div>}
      </div>

      <div className="chat-composer">
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              void handleSend();
            }
          }}
          placeholder="اكتب سؤالك الطبي هنا"
          disabled={isLoading}
        />
        <button type="button" className="btn" onClick={() => void handleSend()} disabled={isLoading}>
          إرسال
        </button>
      </div>
    </div>
  );
}
