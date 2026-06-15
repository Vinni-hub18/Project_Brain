import { useState } from "react";
import api from "../services/api";
import { extractApiErrorMessage } from "../utils/helpers";

function Chatbot() {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hello. I can help explain scan-related information from this app, but I cannot provide a final medical diagnosis. Please consult a doctor for clinical decisions.",
    },
  ]);

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    const userMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setSending(true);

    try {
      const response = await api.post("/chat/", { message: trimmed });
      const reply =
        response?.data?.reply ||
        "I could not generate a response right now. Please try again.";

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: reply },
      ]);
    } catch (error) {
      console.error("Chat request failed:", error?.response?.data || error.message);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: extractApiErrorMessage(
            error,
            "Sorry, the chatbot is unavailable right now."
          ),
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await sendMessage();
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Medical Support Chatbot</h1>
          <p className="section-text">
            Ask about uploaded scans, analysis terminology, or general next-step guidance.
          </p>
        </div>
      </div>

      <div className="chat-layout">
        <section className="panel chat-panel">
          <div className="chat-disclaimer">
            This assistant is for support only and must not be treated as a final diagnosis.
          </div>

          <div className="chat-messages">
            {messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`chat-bubble ${message.role === "user" ? "user" : "assistant"}`}
              >
                <span className="chat-role">
                  {message.role === "user" ? "You" : "Assistant"}
                </span>
                <p>{message.content}</p>
              </div>
            ))}

            {sending && (
              <div className="chat-bubble assistant">
                <span className="chat-role">Assistant</span>
                <p>Typing...</p>
              </div>
            )}
          </div>

          <form className="chat-form" onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="Ask a question about the scan or result..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="chat-input"
            />
            <button type="submit" className="btn btn-primary" disabled={sending}>
              {sending ? "Sending..." : "Send"}
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}

export default Chatbot;