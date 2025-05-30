import React, { useState } from "react";
import "./StreamingAnswer.css";

const StreamingAnswer: React.FC = () => {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [summaryResponse, setSummaryResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://localhost:9000/chat/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ human_message_content: input }),
      });

      if (!res.ok) throw new Error("API Error");

      const reader = res.body?.getReader();
      const decoder = new TextDecoder("utf-8");

      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader!.read();
        done = doneReading;
        const chunk = decoder.decode(value || new Uint8Array(), {
          stream: true,
        });
        setResponse((prev) => prev + chunk);
      }
    } catch (error) {
      console.error(error);
      setResponse("Something went wrong!");
    } finally {
      setLoading(false);
      setInput(""); // Clear input field
    }
  };

  const handleSummary = async () => {
    setSummaryLoading(true);
    setSummaryResponse("");

    try {
      const res = await fetch("http://localhost:9000/summary/", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!res.ok) throw new Error("API Error");

      const reader = res.body?.getReader();
      const decoder = new TextDecoder("utf-8");

      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader!.read();
        done = doneReading;
        const chunk = decoder.decode(value || new Uint8Array(), {
          stream: true,
        });
        setSummaryResponse((prev) => prev + chunk);
      }
    } catch (error) {
      console.error(error);
      setSummaryResponse("Something went wrong!");
    } finally {
      setSummaryLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSubmit();
  };

  return (
    <div className="streaming-container">
      <h1 className="streaming-heading">Streaming LLM Answer</h1>

      <div className="streaming-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask anything..."
        />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Loading..." : "Submit"}
        </button>
      </div>
      <div className="output-contianer">
        <div className="streaming-output">
          {response && (
            <div className="streaming-output-container">{response}</div>
          )}
        </div>

        <div className="summary-container">
          <div className="summary-button-container">
            <button onClick={handleSummary} disabled={summaryLoading}>
              {summaryLoading ? "Loading..." : "Get Summary"}
            </button>
          </div>
          <div className="summary-content">
            {summaryResponse && <div className="streaming-summary-output-container">
              {summaryResponse}
            </div>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StreamingAnswer;
