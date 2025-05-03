// Example frontend implementation for enhanced streaming
import React, { useState, useEffect, useRef } from 'react';

type MessageType = 'step' | 'source' | 'answer' | 'sources_summary' | 'error';

interface StreamMessage {
  type: MessageType;
  content: any; // Could be string or object depending on type
}

export function QueryComponent() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [steps, setSteps] = useState<string[]>([]);
  const [sources, setSources] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setAnswer('');
    setSteps([]);
    setSources([]);
    setError('');
    
    try {
      await queryRagSystem(
        query, 
        (text) => setAnswer(text),
        (step) => setSteps(prev => [...prev, step]),
        (sourceData) => setSources(prev => [...prev, sourceData]),
        (error) => setError(error)
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-container">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Ask'}
        </button>
      </form>
      
      {steps.length > 0 && (
        <div className="steps-container">
          <h4>Processing Steps:</h4>
          <ul>
            {steps.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
        </div>
      )}
      
      {answer && (
        <div className="answer-container">
          <h3>Answer:</h3>
          <div>{answer}</div>
        </div>
      )}
      
      {sources.length > 0 && (
        <div className="sources-container">
          <h4>Sources:</h4>
          <ul>
            {sources.map((source, i) => (
              <li key={i}>
                <div>
                  <strong>Source {i+1}:</strong> 
                  {source.metadata?.title || source.metadata?.source || 'Unknown source'}
                </div>
                <div className="source-snippet">{source.content}</div>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {error && <div className="error-message">Error: {error}</div>}
    </div>
  );
}

export async function queryRagSystem(
  query: string,
  onAnswerUpdate: (text: string) => void,
  onStepUpdate: (step: string) => void,
  onSourceUpdate: (source: any) => void,
  onError: (error: string) => void
): Promise<void> {
  let accumulatedAnswer = '';

  try {
    const response = await fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ question: query })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep the last incomplete line in the buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const content = line.slice(6); // Remove 'data: ' prefix
          
          // Check for completion message
          if (content === '[DONE]') {
            return;
          }

          try {
            // Parse the JSON content
            const jsonContent = JSON.parse(content) as StreamMessage;
            
            // Handle different message types
            switch (jsonContent.type) {
              case 'step':
                onStepUpdate(jsonContent.content);
                break;
                
              case 'source':
                onSourceUpdate(jsonContent.content);
                break;
                
              case 'answer':
                accumulatedAnswer += jsonContent.content;
                onAnswerUpdate(accumulatedAnswer);
                break;
                
              case 'sources_summary':
                // Optional: handle the complete sources summary at the end
                // This could replace the individual sources if preferred
                break;
                
              case 'error':
                onError(jsonContent.content);
                break;
                
              default:
                // For backward compatibility, handle messages without a type
                if (typeof jsonContent.content === 'string') {
                  accumulatedAnswer += jsonContent.content;
                  onAnswerUpdate(accumulatedAnswer);
                }
            }
          } catch (e) {
            console.error('Error parsing SSE message:', e);
          }
        }
      }
    }

  } catch (error) {
    console.error('Error querying RAG system:', error);
    onError(error instanceof Error ? error.message : String(error));
  }
} 