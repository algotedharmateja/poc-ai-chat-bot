import React, { useState } from 'react';

const inputColumns = [
  'Fund Name',
  'Client Account Number',
  'Effective Date',
  'Capital Called ($)',
  'Remaining Commitment ($)',
];

const requiredColumns = [
  'Fund name',
  'Client Account #',
  'Effective Date',
];

const DragDropMapping: React.FC = () => {
  const [mappedCols, setMappedCols] = useState<{ [key: string]: string | null }>(
    () => requiredColumns.reduce((acc, col) => ({ ...acc, [col]: null }), {})
  );

  const handleDragStart = (e: React.DragEvent, value: string) => {
    e.dataTransfer.setData('text/plain', value);
  };

  const handleDrop = (e: React.DragEvent, target: string) => {
    e.preventDefault();
    const droppedValue = e.dataTransfer.getData('text/plain');

    // Prevent same value from being assigned to multiple fields
    if (Object.values(mappedCols).includes(droppedValue)) return;

    setMappedCols((prev) => ({
      ...prev,
      [target]: droppedValue,
    }));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleUndo = (target: string) => {
    setMappedCols((prev) => ({
      ...prev,
      [target]: null,
    }));
  };

  return (
    <div style={{ display: 'flex', gap: '60px', padding: '20px' }}>
      {/* Input Columns (left) */}
      <div>
        <h3>Input Columns</h3>
        {inputColumns.map((col) => {
          const isMapped = Object.values(mappedCols).includes(col);
          return (
            <div
              key={col}
              draggable={!isMapped}
              onDragStart={(e) => handleDragStart(e, col)}
              style={{
                marginBottom: '6px',
                // color: isMapped ? '#9ca3af' : '#1f2937',
                fontStyle: isMapped ? 'italic' : 'normal',
                cursor: isMapped ? 'not-allowed' : 'grab',
              }}
            >
              {col}
            </div>
          );
        })}
      </div>

      {/* Required Columns (right) */}
      <div>
        <h3>Required Columns</h3>
        {requiredColumns.map((req) => (
          <div
            key={req}
            onDrop={(e) => handleDrop(e, req)}
            onDragOver={handleDragOver}
            style={{
              marginBottom: '10px',
              padding: '4px 0',
              // borderBottom: '1px dashed #ccc',
              minHeight: '28px',
              fontSize: '16px',
            }}
          >
            {mappedCols[req] ? (
              <>
                <strong>{mappedCols[req]}</strong>{' '}
                <button
                  onClick={() => handleUndo(req)}
                  style={{
                    marginLeft: '10px',
                    fontSize: '12px',
                    color: '#ef4444',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                >
                  Undo
                </button>
              </>
            ) : (
              <span style={{ color: '#9ca3af' }}>{req}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default DragDropMapping;
