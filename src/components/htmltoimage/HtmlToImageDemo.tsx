// HtmlToImageDemo.js
import React, { useRef } from "react";
import { toPng } from "html-to-image";

const HtmlToImageDemo = () => {
  const domEl = useRef(null);

  const handleDownloadImage = () => {
    if (domEl.current === null) return;

    toPng(domEl.current)
      .then((dataUrl) => {
        const link = document.createElement("a");
        link.download = "my-image.png";
        link.href = dataUrl;
        link.click();
      })
      .catch((err) => {
        console.error("Failed to generate image:", err);
      });
  };

  return (
    <div className="p-4 text-center">
      <div
        ref={domEl}
        style={{
          padding: "20px",
          backgroundColor: "#fef3c7",
          borderRadius: "10px",
          minHeight: "200px", // <--- Add this
          margin: "auto",
          fontFamily: "Arial",
          overflow: "visible", // Optional
        }}
      >
        <h2>Hello from React!</h2>
        <p>This HTML block will be converted into an image.</p>
      </div>

      <button
        onClick={handleDownloadImage}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          backgroundColor: "#10b981",
          color: "#fff",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
        }}
      >
        Download as Image
      </button>
    </div>
  );
};

export default HtmlToImageDemo;
