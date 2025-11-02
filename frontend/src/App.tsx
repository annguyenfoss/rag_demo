import React, { useCallback, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './markdown.css';

type ChatMessage = {
  role: 'user' | 'assistant';
  text: string;
};

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);
  

  const sessionId = useMemo(() => {
    const key = 'rag_session_id';
    let id = sessionStorage.getItem(key);
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem(key, id);
    }
    return id;
  }, []);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [
    input,
    loading,
  ]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text) return;

    setLoading(true);
    setInput('');
    setMessages((m) => [...m, { role: 'user', text }]);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { answer: string };

      setMessages((m) => [...m, { role: 'assistant', text: data.answer }]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessages((m) => [
        ...m,
        { role: 'assistant', text: `Error: ${msg}` },
      ]);
    } finally {
      setLoading(false);
      // scroll to bottom
      listRef.current?.lastElementChild?.scrollIntoView({
        behavior: 'smooth',
        block: 'end',
      });
    }
  }, [input]);

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>RAG Chatbot</h1>
        <div style={styles.messages} ref={listRef}>
          {messages.length === 0 ? (
            <div style={styles.hint}>Ask a question to get started.</div>
          ) : (
            messages.map((m, i) => (
              <div key={i} style={
                m.role === 'user' ? styles.userMsg : styles.assistantMsg
              }>
                <div style={styles.role}>{m.role}</div>
                {m.role === 'assistant' ? (
                  <div className="markdownContent">
                    <ReactMarkdown>{m.text}</ReactMarkdown>
                  </div>
                ) : (
                  <div style={styles.textContent}>{m.text}</div>
                )}
              </div>
            ))
          )}
        </div>
        <div style={styles.spacer} />
      </div>
      <div style={styles.inputContainer}>
        <div style={styles.inputRow}>
          <input
            style={styles.input}
            placeholder={loading ? 'Thinking…' : 'Type your message'}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && canSend) {
                e.preventDefault();
                void send();
              }
            }}
            disabled={loading}
          />
          <button style={styles.button} onClick={() => void send()} disabled={!canSend}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: '#f5f5f7',
    color: '#1d1d1f',
    fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
  },
  container: {
    flex: 1,
    maxWidth: '100%',
    padding: '80px 60px 20px 60px',
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  title: { 
    margin: '0 0 20px 0', 
    fontSize: 24, 
    fontWeight: 700, 
    color: '#1d1d1f',
  },
  messages: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    background: 'transparent',
  },
  userMsg: {
    alignSelf: 'flex-end',
    background: '#007AFF',
    color: '#ffffff',
    padding: '10px 14px',
    borderRadius: 18,
    maxWidth: '85%',
  },
  assistantMsg: {
    alignSelf: 'flex-start',
    background: '#e9e9eb',
    color: '#1d1d1f',
    padding: '10px 14px',
    borderRadius: 18,
    maxWidth: '85%',
    lineHeight: '1.6',
  },
  role: { fontSize: 14, opacity: 0.6, marginBottom: 4 },
  textContent: {
    fontSize: 18,
    whiteSpace: 'pre-wrap',
  },
  hint: { opacity: 0.5, fontSize: 18, color: '#86868b' },
  spacer: { height: 20 },
  inputContainer: {
    position: 'sticky',
    bottom: 0,
    background: '#f5f5f7',
    borderTop: '1px solid #e5e5e7',
    padding: '16px 60px 80px 60px',
    marginTop: 'auto',
  },
  inputRow: {
    display: 'flex',
    gap: 8,
    maxWidth: '100%',
  },
  input: {
    flex: 1,
    borderRadius: 20,
    border: '1px solid #d1d1d6',
    padding: '12px 16px',
    background: '#ffffff',
    color: '#1d1d1f',
    fontSize: 18,
  },
  button: {
    borderRadius: 20,
    border: 'none',
    padding: '12px 20px',
    background: '#007AFF',
    color: '#ffffff',
    cursor: 'pointer',
    fontWeight: 600,
  },
};
