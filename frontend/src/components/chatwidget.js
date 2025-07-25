import React, { useState, useRef, useEffect } from "react";
import { IoMdChatboxes } from "react-icons/io";
import { MdClose } from "react-icons/md";
import axios from "axios";

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

 const sendMessage = async () => {
  if (!message.trim() || loading) return;

  setLoading(true);
  const newUserMessage = { sender: "user", text: message };
  setChatHistory((prev) => [...prev, newUserMessage]);

  try {
    const res = await axios.post(
      "http://localhost:8000/api/chat",
      { message}, 
      {
        headers: {
          "Content-Type": "application/json", 
        },
      }
    );

    const botReply = res.data.response;
    setChatHistory((prev) => [...prev, { sender: "bot", text: botReply }]);
  } catch (err) {
    console.error("API call failed:", err.response?.data || err.message || err);
    setChatHistory((prev) => [
      ...prev,
      { sender: "bot", text: "⚠️ Could not connect to the chatbot." },
    ]);
  }

  setMessage("");
  setLoading(false);
};



  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  return (
    <div style={styles.widgetContainer}>
      {isOpen ? (
        <div style={styles.chatWindow}>
          <div style={styles.header}>
            <span>Chat With AI 🤖</span>
            <MdClose style={styles.icon} onClick={() => setIsOpen(false)} />
          </div>

          <div style={styles.chatBody}>
            <div style={styles.chatBox}>
              {chatHistory.map((msg, index) => (
                <div
                  key={index}
                  style={{
                    ...styles.message,
                    alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                    backgroundColor: msg.sender === "user" ? "#DCF8C6" : "#e6e6e6",
                  }}
                >
                  <b>{msg.sender === "user" ? "You" : "Bot"}:</b> {msg.text}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>

            <div style={styles.inputArea}>
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask something..."
                style={styles.input}
                disabled={loading}
              />
              <button
                style={styles.button}
                onClick={sendMessage}
                disabled={loading}
              >
                {loading ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <button style={styles.fab} onClick={() => setIsOpen(true)}>
          <IoMdChatboxes size={28} />
        </button>
      )}
    </div>
  );
};

const styles = {
  widgetContainer: {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    zIndex: 9999,
  },
  fab: {
    backgroundColor: "#6c2bd9",
    color: "#fff",
    border: "none",
    borderRadius: "50%",
    width: "60px",
    height: "60px",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
  },
  chatWindow: {
    width: "370px",
    height: "500px",
    backgroundColor: "#fff",
    borderRadius: "12px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  header: {
    backgroundColor: "#6c2bd9",
    color: "#fff",
    padding: "10px 15px",
    fontWeight: "bold",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  icon: {
    cursor: "pointer",
  },
  chatBody: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
  },
  chatBox: {
    flex: 1,
    overflowY: "auto",
    padding: "10px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    background: "#f9f9f9",
  },
  message: {
    padding: "8px 12px",
    borderRadius: "10px",
    maxWidth: "70%",
    fontSize: "14px",
  },
  inputArea: {
    display: "flex",
    padding: "10px",
    borderTop: "1px solid #ccc",
    backgroundColor: "#fff",
  },
  input: {
    flex: 1,
    padding: "10px",
    fontSize: "14px",
    border: "1px solid #ccc",
    borderRadius: "6px",
    outline: "none",
  },
  button: {
    marginLeft: "10px",
    padding: "10px 16px",
    backgroundColor: "#6c2bd9",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },
};

export default ChatWidget;
