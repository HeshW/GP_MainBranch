import React, { useState } from 'react';
import { postChat } from '@/api/client';
import { ChatMessage } from '@/types';

export function ChatInterface() {
  const [chatSession] = useState<string>("session-" + Math.random().toString(36).substring(7));
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);

  const handleChat = async () => {
    if (!chatInput.trim()) return;
    const msg = chatInput.trim();
    setChatInput("");
    setChatMessages(prev => [...prev, { role: 'user', content: msg }]);
    setChatLoading(true);
    try {
      const data = await postChat({ session_id: chatSession, message: msg }) as any;
      setChatMessages(prev => [...prev, { role: 'model', content: data.response }]);
    } catch (e) {
      setChatMessages(prev => [...prev, { role: 'model', content: "Error: " + String(e) }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="chat-section" style={{ direction: "rtl", textAlign: "right" }}>
      <h3 style={{ marginBottom: "1rem", color: "#333" }}>طبيبك الرقمي - اسأل عن حالتك:</h3>
      
      <div className="chat-history" style={{ border: "1px solid #d1d5db", padding: "1rem", minHeight: "150px", maxHeight: "300px", overflowY: "auto", marginBottom: "1rem", borderRadius: "8px", background: "#f9fafb" }}>
        {chatMessages.length === 0 ? (
          <p style={{ color: "#6b7280", textAlign: "center", margin: "2rem 0" }}>الدردشة فارغة. يمكنك سؤالي عن نتائج التشخيص أو خطة العلاج المذكورة أعلاه.</p>
        ) : (
          chatMessages.map((msg, i) => (
            <div key={i} style={{ marginBottom: "1rem", padding: "0.75rem", background: msg.role === "user" ? "#dbeafe" : "#ffffff", color: "#1f2937", border: "1px solid #e5e7eb", borderRadius: "8px", maxWidth: "85%", marginInlineStart: msg.role === "user" ? "auto" : "0", boxShadow: "0 1px 2px rgba(0,0,0,0.05)" }}>
              <strong style={{ display: "block", marginBottom: "0.25rem", color: msg.role === "user" ? "#1d4ed8" : "#059669" }}>{msg.role === "user" ? "أنت:" : "الطبيب الرقمي:"}</strong>
              <span style={{ whiteSpace: "pre-wrap", lineHeight: "1.5" }}>{msg.content}</span>
            </div>
          ))
        )}
        {chatLoading && <div style={{ color: "#6b7280", fontStyle: "italic", marginTop: "0.5rem" }}>جاري الكتابة...</div>}
      </div>
      
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <input 
          type="text" 
          value={chatInput} 
          onChange={e => setChatInput(e.target.value)} 
          onKeyDown={e => e.key === 'Enter' && handleChat()}
          placeholder="اكتب رسالتك للطبيب الرقمي هنا..." 
          style={{ flex: 1, padding: "0.75rem", borderRadius: "6px", border: "1px solid #d1d5db", fontSize: "1rem" }} 
          disabled={chatLoading}
        />
        <button 
          onClick={handleChat} 
          disabled={chatLoading} 
          className="btn" 
          style={{ padding: "0 1.5rem", borderRadius: "6px" }}
        >
          إرسال
        </button>
      </div>
    </div>
  );
}
