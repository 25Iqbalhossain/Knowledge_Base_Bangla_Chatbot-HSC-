import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api",
  withCredentials: false,
});

const ChatBox = () => {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const sendMessage = async () => {
    if (!message.trim()) return;

    const newChat = { sender: "user", text: message };
    setChatHistory((prev) => [...prev, newChat]);

    try {
      const res = await api.post("/chat", { message }, {
        headers: { "Content-Type": "application/json" },
      });

      const botReply = res.data.response;
      setChatHistory((prev) => [...prev, { sender: "bot", text: botReply }]);
    } catch (err) {
      console.error("Chatbot error:", err?.response?.data || err.message);
      setChatHistory((prev) => [
        ...prev,
        { sender: "bot", text: "Error connecting to chatbot." },
      ]);
    }

    setMessage("");
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.chatContainer}>
        <h2 style={styles.header}>🤖 Bangla AI Assistant</h2>
        <div style={styles.chatBox}>
          {chatHistory.map((msg, i) => (
            <div
              key={i}
              style={{
                ...styles.message,
                alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                backgroundColor: msg.sender === "user" ? "#96f2d7" : "#e0e0e0",
                color: msg.sender === "user" ? "#003e1f" : "#222",
                borderTopLeftRadius: msg.sender === "user" ? "20px" : "5px",
                borderTopRightRadius: msg.sender === "user" ? "5px" : "20px",
              }}
            >
              <span style={styles.senderIcon}>
                {msg.sender === "user" ? "🧑‍🎓" : "🤖"}
              </span>
              <span>{msg.text}</span>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
        <div style={styles.inputArea}>
          <input
            style={styles.input}
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="PLease type your query ..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button style={styles.button} onClick={sendMessage}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

const styles = {
  wrapper: {
    backgroundColor: "#f4f6fb",
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "30px",
  },
  chatContainer: {
    width: "100%",
    maxWidth: "700px",
    backgroundColor: "#101010",
    padding: "24px",
    borderRadius: "18px",
    boxShadow: "0 8px 20px rgba(0,0,0,0.5)",
    color: "#fff",
    fontFamily: "Segoe UI, sans-serif",
  },
  header: {
    textAlign: "center",
    marginBottom: "20px",
    color: "#00ff94",
    fontSize: "22px",
  },
  chatBox: {
    height: "420px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    marginBottom: "18px",
    padding: "16px",
    border: "1px solid #333",
    borderRadius: "14px",
    background: "#1f1f1f",
  },
  message: {
    padding: "12px 18px",
    borderRadius: "20px",
    maxWidth: "70%",
    wordBreak: "break-word",
    boxShadow: "0 2px 6px rgba(0,0,0,0.2)",
    fontSize: "15px",
    lineHeight: "1.5",
    display: "inline-block",
  },
  senderIcon: {
    fontWeight: "bold",
    marginRight: "8px",
  },
  inputArea: {
    display: "flex",
    gap: "10px",
    alignItems: "center",
  },
  input: {
    flex: 1,
    padding: "14px 16px",
    fontSize: "16px",
    borderRadius: "10px",
    border: "1px solid #444",
    backgroundColor: "#2c2c2c",
    color: "#fff",
  },
  button: {
    padding: "14px 20px",
    backgroundColor: "#00ff94",
    color: "#000",
    border: "none",
    borderRadius: "10px",
    fontWeight: "bold",
    cursor: "pointer",
    transition: "0.3s ease-in-out",
  },
};

export default ChatBox;
