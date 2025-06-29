import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/answer', {
        question: question
      });
      setAnswer(response.data.answer);
    } catch (err) {
      console.error(err);
      setAnswer('Error retrieving answer.');
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>RAG QA App</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question"
          style={{ width: '300px' }}
        />
        <button type="submit">Ask</button>
      </form>
      <div style={{ marginTop: '20px' }}>
        <strong>Answer:</strong>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;
