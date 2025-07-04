import React, { useState, useEffect} from "react";
import "./StreamingAnswer.css";
import FileUpload from './FileUpload';

const StreamingAnswer: React.FC = () => {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [summaryResponse, setSummaryResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const [citationQuery, setCitationQuery] = useState("");
  const [citationResponse, setCitationResponse] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [citationLoading, setCitationLoading] = useState(false);

  const [notifications, setNotifications] = useState<string[]>([]);

  useEffect(() => {
    const eventSource = new EventSource("http://localhost:9000/sse"); // adjust URL if needed

    eventSource.onmessage = (event) => {
      // Append new notification to list
      setNotifications((prev) => [...prev, event.data]);
    };

    eventSource.onerror = (err) => {
      console.error("SSE connection error:", err);
      eventSource.close(); // Optionally close on error
    };

    // Cleanup on unmount
    return () => {
      eventSource.close();
    };
  }, []);



  const handleCitationQuery = async () => {
    if (!citationQuery.trim()) return;

    setCitationLoading(true);
    setCitationResponse("");
    setCitations([]);

    try {
      const res = await fetch("http://localhost:9000/stream-collection-agent/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          collection_name: "docfast",
          query: citationQuery,
        }),
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

        const lines = chunk.split("\n").filter(Boolean);
        for (const line of lines) {
          const parsed = JSON.parse(line);
          if (parsed.type === "output_chunk") {
            setCitationResponse((prev) => prev + parsed.chunk);
          } else if (parsed.type === "citation") {
            const citationText = `${parsed.citation.content} (Score: ${parsed.citation.score})`;
            setCitations((prev) => [...prev, citationText]);
          }
        }
      }
    } catch (error) {
      console.error(error);
      setCitationResponse("Something went wrong!");
    } finally {
      setCitationLoading(false);
      setCitationQuery("");
    }
  };


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

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("collection_name", "ui_collection");

    try {
      const res = await fetch("http://localhost:9000/upload-file/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("File upload failed");

      const data = await res.json();
      console.log(data.message);
      alert("File uploaded and indexed successfully!");
    } catch (error) {
      console.error("Error during file upload:", error);
      alert("Something went wrong with the file upload");
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSubmit();
  };

  return (
    <>
    <div className="streaming-container">
      <h1 className="streaming-heading">Streaming LLM Answer</h1>

      <div className="notifications-container">
        <h2>Notifications</h2>
        <ul>
          {notifications.map((note, index) => (
            <li key={index}>{note}</li>
          ))}
        </ul>
      </div>

      <div className="streaming-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask anything..." />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Loading..." : "Submit"}
        </button>
      </div>

      <div className="output-contianer">
        <div className="streaming-output">
          {response && <div className="streaming-output-container">{response}</div>}
        </div>

        <div className="summary-container">
          <button onClick={handleSummary} disabled={summaryLoading}>
            {summaryLoading ? "Loading..." : "Get Summary"}
          </button>
          <div className="summary-content">
            {summaryResponse && (
              <div className="streaming-summary-output-container">{summaryResponse}</div>
            )}
          </div>
        </div>
      </div>

      <div className="summary-container">
        <div>
          <p>Upload file for indexing</p>
          <FileUpload onFileSelect={handleFileSelect} onUpload={handleUpload} />
        </div>
      </div>
    </div><div className="citation-agent-container">
        <h2>Citation Agent Query</h2>
        <div className="citation-agent-input">
          <input
            type="text"
            value={citationQuery}
            onChange={(e) => setCitationQuery(e.target.value)}
            placeholder="Ask a question about indexed files..." />
          <button onClick={handleCitationQuery} disabled={citationLoading}>
            {citationLoading ? "Loading..." : "Submit"}
          </button>
        </div>

        <div className="citation-output-wrapper">
          <div className="citation-output-box">
            <h3>Answer</h3>
            <div className="citation-output">{citationResponse}</div>
          </div>
          <div className="citation-output-box">
            <h3>Citations</h3>
            <div className="citation-citations">
              {citations.length === 0 && <p>No citations yet</p>}
              <ul>
                {citations.map((citation, index) => (
                  <li key={index}>{citation}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div></>


    
  );
};

export default StreamingAnswer;
