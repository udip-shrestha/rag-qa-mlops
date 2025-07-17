import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);


  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [uploading, setUploading] = useState(false);



    // Inject keyframes once
    useEffect(() => {
      const styleTag = document.createElement('style');
      styleTag.innerHTML = `
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `;
      document.head.appendChild(styleTag);
    }, []);
  
    const spinnerStyle = {
      marginTop: "10px",
      width: "30px",
      height: "30px",
      border: "4px solid #ccc",
      borderTop: "4px solid #333",
      borderRadius: "50%",
      animation: "spin 1s linear infinite"
    };


  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);         // start loading
    setAnswer('');            // clear previous answer
    try {
      const response = await axios.post('/answer', {
        question: question
      });
      
      setAnswer(response.data.answer);
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        setAnswer("Error: " + err.response.data.detail);
      } else {
        setAnswer("Error retrieving answer.");
      }
    } finally {
      setLoading(false);      // stop loading
    }
  };


  const handleUpload = async () => {
    if (!file) return;
  
    setUploading(true);
    setUploadStatus("");
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const response = await axios.post("http://localhost:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
  
      setUploadStatus(`Uploaded ${response.data.file} successfully.`);
    } catch (err) {
      console.error(err);
      setUploadStatus("Upload failed.");
    } finally {
      setUploading(false);
    }
  };
  
  
  
  

  return (
    <div style={{
      fontFamily: 'sans-serif',
      maxWidth: '600px',
      margin: '50px auto',
      padding: '20px',
      border: '1px solid #ddd',
      borderRadius: '12px',
      boxShadow: '0 0 10px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '20px' }}>RAG QA App</h1>
  
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question"
          style={{
            flex: 1,
            padding: '10px',
            borderRadius: '6px',
            border: '1px solid #ccc'
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: '10px 20px',
            borderRadius: '6px',
            border: 'none',
            backgroundColor: loading ? '#888' : '#007bff',
            color: 'white',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>
  
      <div style={{ marginTop: '30px' }}>
        <strong>Answer:</strong>
        <div>
          {loading ? <div style={spinnerStyle}></div> : <p>{answer}</p>}
        </div>
      </div>

      <div style={{ marginTop: '30px' }}>
      <h3>Upload Document</h3>
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
        <input type="file" accept=".txt" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} disabled={uploading}>
          {uploading ? "Uploading..." : "Upload"}
        </button>
      </div>
      {uploadStatus && <p style={{ marginTop: "10px", color: "green" }}>{uploadStatus}</p>}
    </div>




      

    </div>
  );
  
}

export default App;



