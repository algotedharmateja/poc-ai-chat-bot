import React, { useState } from "react";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onUpload: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, onUpload }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleChange} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default FileUpload;
